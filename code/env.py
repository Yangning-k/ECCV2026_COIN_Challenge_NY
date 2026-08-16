import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from utils import GeminiLLM, ClientBasedLLM
from retrying import retry
import time

_MAX_TIME_ALLOWED = 600  # in seconds

ANSWER_PROMPT = "You are a faithful assistant. Answer correnctly to the following question based on the above image: {QUESTION}. Be concise (under 15 words)."


def _validate_action(action):
    assert (
        action["question"] is None
        or action["conclusion"] is None
        and not (action["question"] is None and action["conclusion"] is None)
    ), (
        "Wrong action format: one among 'question' and 'conclusion' must be None, but not both"
    )


class MockOracle:
    def __init__(self):
        pass

    def ask(*, prompt="", images=[]):
        return "Yes that is true [Mock answer]"


class QAEnv(gym.Env):
    """
    A gym environment that represents the QA game.
    """

    metadata = {"render_modes": ["rgb"]}

    def __init__(
        self,
        client: Union[GeminiLLM, Any],
        jsonl_path: str,
        render_mode: Optional[str] = None,
        image_width: int = 512,
        image_height: int = 512,
        max_steps: int = 60,
        task_type: str = "category",
    ):
        """_summary_

        Args:
            client (_type_): oracle LLM client
            jsonl_path (str): path of the JSONL containing info about the episodes
            render_mode (Optional[str], optional): Render mode. Defaults to None.
            image_width (int, optional): image (observation) width. Defaults to 512.
            image_height (int, optional): image (observation) height. Defaults to 512.
        """
        super().__init__()

        self.oracle_client = client
        self.jsonl_path = Path(jsonl_path)
        self.render_mode = render_mode
        self.image_height = image_height
        self.image_width = image_width
        self.max_steps = max_steps
        self.task_type = task_type

        # Load all episodes from JSONL file
        self.episodes = self._load_episodes()
        self.current_episode_idx = 0
        self.current_step = 0
        self.n_questions = 0
        self.current_episode_data = None

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.image_height,
                    self.image_width,
                    3,
                ),  # Height x Width x Channels (RGB)
                dtype=np.uint8,
            ),
            "answer": spaces.Text(max_length=300),
        })
        self.action_space = spaces.Dict({
            "question": spaces.Text(max_length=300),
            "conclusion": spaces.Discrete(2),  # 0 if No, 1 if Yes
        })

    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load all episodes from the JSONL file."""
        episodes = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                episode_data = json.loads(line.strip())
                episodes.append(episode_data)
        return episodes

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and load the next episode."""
        super().reset(seed=seed)
        self.initial_time = time.time()
        self.n_questions = 0
        self.n_successes = 0

        # Select episode (sequential or random based on options)
        if options and options.get("episode_idx") is not None:
            self.current_episode_idx = options["episode_idx"]
        else:
            # Sequential by default, wrap around at the end
            self.current_episode_idx = self.current_episode_idx % len(self.episodes)

        # Load the episode data

        self.current_episode_data = self.episodes[self.current_episode_idx]

        self.current_step = 0

        self.current_target_image = Image.open(
            self._resolve_image_path(self.current_episode_data["path"])
        )
        # TODO select type of task here here
        self.current_task = self.current_episode_data["tasks"][self.task_type]
        self.current_distractor_idx = 0
        self.distractors = self.current_episode_data["distractors"]

        observation = self._get_observation(question=None)
        info = self._get_info()
        info.update({"category": self.current_episode_data["tasks"]["category"]})

        # Move to next episode for the next reset
        self.current_episode_idx += 1

        return observation, info

    def step(
        self, action: dict
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""

        _validate_action(action)

        self.current_step += 1

        # We compute this before the 'switching image logic' otherwise it will mess them up
        # (but the observation needs to be computed after)
        terminated = self._is_terminated(action)
        truncated = self._is_truncated()
        reward = self._compute_reward(action)
        info = self._get_info()

        # If the conclusion is right we move forward
        if (
            action["conclusion"] is not None
            and bool(action["conclusion"])
            == (self.distractors[self.current_distractor_idx]["match"])
            and not (terminated or truncated)
        ):
            self.current_distractor_idx += 1
            info.update({"new_distractor": True})

        observation = self._get_observation(action["question"])

        return observation, reward, terminated, truncated, info

    @retry(stop_max_attempt_number=5, wait_fixed=80000)
    def ask_oracle(self, prompt, image, description=None):
        return self.oracle_client.ask(
            prompt=prompt,
            images=[image],
        )

    def _get_observation(self, question: str = None) -> np.ndarray:
        """Extract observation from current episode data and step."""
        prompt_to_use = ANSWER_PROMPT.format(QUESTION=question)
        answer = None
        if question is not None:
            answer = self.ask_oracle(prompt_to_use, self.current_target_image)
        image_path = self._resolve_image_path(
            self.distractors[self.current_distractor_idx]["path"]
        )
        return dict(
            image=np.array(Image.open(image_path)),
            image_path=str(image_path),
            answer=answer,
        )

    def _resolve_image_path(self, path: str) -> Path:
        raw = Path(path)
        if raw.is_file():
            return raw
        search = [Path.cwd() / raw]
        if self.jsonl_path.is_file():
            search.append(self.jsonl_path.parent / raw)
            search.append(self.jsonl_path.parent.parent / raw)
        for candidate in search:
            if candidate.is_file():
                return candidate
        return raw

    def _compute_reward(self, action: dict) -> float:
        """Compute reward for the current step."""

        question = action["question"]
        conclusion = action["conclusion"]

        if conclusion is not None:
            # 1: true, 0: false
            conclusion = bool(conclusion)
            # Whether the ansdwer is right or no (i.e. it maches or not)
            matching = (
                conclusion == self.distractors[self.current_distractor_idx]["match"]
            )
            if matching:
                self.n_successes += 1
            return 10 * (1 if matching else -1)
        if question is not None:
            return -1

    def _is_terminated(self, action) -> bool:
        """Check if episode has terminated (reached goal or failure state)."""
        conclusion = action["conclusion"]

        if conclusion is not None:
            # 1: true, 0: false
            conclusion = bool(conclusion)
            # If we have a failure, or we reached the last image
            return conclusion != self.distractors[self.current_distractor_idx][
                "match"
            ] or self.current_distractor_idx == (len(self.distractors) - 1)

        return False

    def _is_truncated(self) -> bool:
        """Check if episode has been truncated (max steps reached or max time)."""
        return (
            self.current_step >= self.max_steps
            or (time.time() - self.initial_time) > _MAX_TIME_ALLOWED
        )

    def _get_info(self) -> Dict[str, Any]:
        """Return additional information about the current state."""
        info = {
            "episode_idx": self.current_episode_idx - 1,
            "step": self.current_step,
            "n_questions": self.n_questions,
            "task_image": self.current_target_image,
            "target_description": self.current_episode_data["tasks"][self.task_type],
            "new_distractor": False,
        }
        return info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # TODO: Implement rendering logic
            print(f"Episode {self.current_episode_idx - 1}, Step {self.current_step}")

    def close(self):
        """Clean up resources."""
        pass
