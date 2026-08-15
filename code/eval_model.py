import numpy as np
import json
from PIL import Image
from utils import GeminiLLM, ClientBasedLLM, load_api_keys
from env import QAEnv
import time
import argparse
import os
from Questioner import QuestionerLocalVLM

ORACLE_MODEL_ID = "gemini-3-flash"
# Default local Oracle: the vLLM server currently exposed on this machine.
DEFAULT_LOCAL_ORACLE_MODEL_ID = (
    "/shared_disk/users/ning.yang/checkpoints/Qwen2.5-VL-32B-Instruct"
)
DEFAULT_LOCAL_ORACLE_PORT = 8000
DEFAULT_QUESTIONER_MODEL_ID = (
    "/shared_disk/users/ning.yang/Codes/coin_challenge/checkpoints/"
    "Qwen2.5-VL-7B-Instruct-tuned-final-questions-v1-2"
)
DEFAULT_QUESTIONER_PORT = 8001
EPISODES_JSONL = "episodes_{run_type}.jsonl"

parser = argparse.ArgumentParser(prog="eval-QA-model")
parser.add_argument("start_idx", type=int, help="First trajectory index")
parser.add_argument("end_idx", type=int, help="Last trajectory index")
parser.add_argument(
    "--description-type",
    default="category",
    help="Type of description. Choose one among 'all', 'category', 'color', 'context', 'color_context_feature', 'color_feature', 'color_context'.",
)
parser.add_argument(
    "--local",
    type=int,
    default=1,
    help="If 1, will use a local VLM (run from VLLM) as oracle; if 0, will use Gemini API as oracle.",
)
parser.add_argument(
    "--prompt-variant",
    default="paper",
    choices=[
        "paper",
        "example",
        "paper_yn",
        "paper_adaptive",
        "our_prompt_v1",
        "our_prompt_v2",
        "our_prompt_v3",
    ],
    help="Questioner prompt: 'paper' = Light-CoNav arXiv:2604.00265; "
    "'paper_yn' = paper + yes/no discriminative-question constraint; "
    "'paper_adaptive' = paper_yn + adaptive ask-vs-conclude rule based on "
    "description information content; "
    "'our_prompt_v1' = Our_Prompt_V1 (OBSERVE/COMPARE/DECIDE CoT, target-profile "
    "questions; see PROMPT_DESIGN_NOTES.md); "
    "'our_prompt_v2' = Our_Prompt_V2 (category-only ask rule and compact "
    "OBSERVE/COMPARE/DECIDE CoT); "
    "'our_prompt_v3' = Our_Prompt_V3 (v1 score boundary with v2 compact "
    "reasoning and stricter attribute/question checks); "
    "'example' = challenge-repo QUESTIONER_EXAMPLE_PROMPT.",
)
parser.add_argument(
    "--policy",
    default="anti_fp",
    choices=["baseline", "anti_fp", "force_ask_short", "uncertainty_gate",
             "dedup_force_decide", "dedup_category_only"],
    help="Decision policy. baseline=raw model; anti_fp=block premature match on "
    "short descriptions + cap questions/obs; force_ask_short=only force-ask; "
    "uncertainty_gate=anti_fp + wording-based confidence gate (ask when the "
    "model sounds unsure despite a firm score).",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Questioner sampling temperature (0 recommended for FR).",
)
parser.add_argument(
    "--tag",
    default="",
    help="Optional tag appended to result filename for experiment tracking.",
)
parser.add_argument(
    "--results-dir",
    default="results",
    help="Directory to write result JSON files.",
)

args = parser.parse_args()

local = args.local

run_type = "train"

ALLOWED_DESCPRITION_TYPES = [
    "all",
    "category",
    "color",
    "context",
    "color_context_feature",
    "color_feature",
    "color_context",
]

assert args.description_type in ALLOWED_DESCPRITION_TYPES, f"--description-type should be one among {ALLOWED_DESCPRITION_TYPES}"



def _add_obs_and_question_to_log(
    obs, new_obs, action, observations, observation_paths, actions, answers, reasonings
):
    image = Image.fromarray(obs["image"])
    image_path = obs.get("image_path")
    if len(observations) == 0:
        observations.append(image)
        observation_paths.append(image_path)
        actions.append([])
        answers.append([])
        reasonings.append([])
    elif image != observations[-1]:
        observations.append(image)
        observation_paths.append(image_path)
        if action["question"] is not None:
            actions.append([f"Q: {action['question'][:200]}"])
            answers.append([f"A: {new_obs['answer']}"])
        else:
            actions.append([f"C: {bool(action['conclusion'])}"])
            answers.append([])
        reasonings.append([action["reasoning"]])
    else:
        if action["question"] is not None:
            assert new_obs["answer"] is not None
            s = f"Q: {action['question'][:200]}"
            answers[-1].append(f"A: {new_obs['answer']}")
        else:
            s = f"C: {bool(action['conclusion'])}"
        actions[-1].append(s)
        reasonings[-1].append(action["reasoning"])


already_done_ids = set()

# Example usage:
if __name__ == "__main__":
    now = str(time.time_ns())

    load_api_keys()
    # model = "gemini-3-flash-preview"

    # -------------------------------------- ORACLE ----------------------------------------
    # if you want to use a custom oracle, you have to change these lines. You can
    # implement it however you want, but it should inher it from the class OracleInterace
    # (see file Oracle.py)
    if not local:
        oracle_client = GeminiLLM(model_id=ORACLE_MODEL_ID, temperature=1e-6)
        print(f"[INFO] Using oracle model: {ORACLE_MODEL_ID}")
    else:
        oracle_model_id = os.environ.get(
            "ORACLE_MODEL_ID", DEFAULT_LOCAL_ORACLE_MODEL_ID
        )
        oracle_port = int(
            os.environ.get(
                "ORACLE_VLLM_PORT",
                os.environ.get("MY_VLLM_PORT", DEFAULT_LOCAL_ORACLE_PORT),
            )
        )
        oracle_client = ClientBasedLLM(
            model_id=oracle_model_id,
            port=oracle_port,
            url=f"http://localhost:{oracle_port}/v1",
        )
        ## Or you can use your oracle here
        # oracle_client = YourOracle
        print(f"[INFO] Using oracle model: {oracle_model_id} (port={oracle_port})")
        # TODO You can also use your oracle here

    print(f"[INFO] Using oracle: {oracle_client}")
    print(f"[INFO] Questioner prompt_variant: {args.prompt_variant}")
    print(
        f"[INFO] Questioner policy={args.policy} temperature={args.temperature} "
        f"tag={args.tag!r}"
    )
    # --------------------------------------------------------------------------------------
    if args.description_type.lower() == "all":
        task_types = ALLOWED_DESCPRITION_TYPES[1:]
    else:
        task_types = [args.description_type]

    episodes_path = os.environ.get(
        "EPISODES_JSONL_OVERRIDE"
    ) or EPISODES_JSONL.format(run_type=run_type)
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(
            f"Episodes file not found: {episodes_path}. "
            "Expected episodes_train.jsonl in the repo root."
        )

    for task_type in task_types:
        env = QAEnv(
            oracle_client,
            episodes_path,
            render_mode="rgb",
            task_type=task_type,
        )

        log_data = dict(
            id=[],
            target_image=[],
            task=[],
            observations=[],
            questions=[],
            answers=[],
            reasonings=[],
            n_successes=[],
            n_questions=[],
            time_required=[],
        )
        for episode in range(args.start_idx, args.end_idx):
            _observations = []
            _observation_paths = []
            _actions = []
            _answers = []
            _reasonings = []
            try:
                old_obs, info = env.reset(options={"episode_idx": episode})
            except IndexError as e:
                continue
            print(f"EPISODE: {episode}, ID: {env.current_episode_data['id']}\n")
            if env.current_episode_data["id"] in already_done_ids:
                print(
                    f"I did already run episode with id '{env.current_episode_data['id']}' for subset '{task_type}'"
                )
                continue
            try:
                _add_obs_and_question_to_log(
                    old_obs,
                    {},
                    {},
                    _observations,
                    _observation_paths,
                    _actions,
                    _answers,
                    _reasonings,
                )
            except Exception as e:  # noqa
                print("[ERROR] Error in episode number: ", episode)
                print(str(e))
                continue

            questioner_model_id = os.environ.get(
                "QUESTIONER_MODEL_ID", DEFAULT_QUESTIONER_MODEL_ID
            )
            questioner_port = int(
                os.environ.get("QUESTIONER_VLLM_PORT", DEFAULT_QUESTIONER_PORT)
            )
            questioner = QuestionerLocalVLM(
                info,
                model_id=questioner_model_id,
                port=questioner_port,
                prompt_variant=args.prompt_variant,
                temperature=args.temperature,
                policy=args.policy,
                description_type=task_type,
            )

            questioner.reset_time()
            print(f"Task is: {info['target_description']}")

            for step in range(200):
                print("=============")
                # action = ACTIONS[step]
                try:
                    action = questioner.ask_or_conclude(old_obs)
                except Exception as e:  # noqa
                    print("[ERROR] Error in episode number: ", episode)
                    print(str(e))
                    break

                print(f"Current action: {action}")
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("new_distractor"):
                    questioner.notify_new_observation()

                if action["question"] is not None:
                    a = ""
                    if obs["answer"] is not None:
                        a = obs["answer"]
                    questioner.add_answer(a)
                try:
                    _add_obs_and_question_to_log(
                        old_obs,
                        obs,
                        action,
                        _observations,
                        _observation_paths,
                        _actions,
                        _answers,
                        _reasonings,
                    )
                except Exception as e:  # noqa
                    print("[ERROR] Error in episode number: ", episode)
                    print(str(e))
                    _add_obs_and_question_to_log(
                        old_obs,
                        obs,
                        action,
                        _observations,
                        _observation_paths,
                        _actions,
                        _answers,
                        _reasonings,
                    )
                    break

                # env.render()

                old_obs = obs
                if terminated or truncated:
                    times = round(time.time() - env.initial_time, 2)
                    try:
                        print(f"Episode finished after {step + 1} steps")
                        _id = env.current_episode_data["id"]
                        _target_image_path = env.current_episode_data["path"]
                        _task = env.current_episode_data["tasks"][task_type]
                        _n_successes = env.n_successes
                        _n_questions = questioner.n_questions
                        _time_required = round(questioner.time_required, 2)

                        # Append (image fields store paths only)
                        log_data["id"].append(_id)
                        log_data["target_image"].append(_target_image_path)
                        log_data["task"].append(_task)
                        log_data["n_successes"].append(_n_successes)
                        log_data["n_questions"].append(_n_questions)
                        log_data["time_required"].append(_time_required)
                        log_data["observations"].append(list(_observation_paths))
                        log_data["questions"].append(_actions)
                        log_data["answers"].append(_answers)
                        log_data["reasonings"].append(_reasonings)
                    except Exception as e:  # noqa
                        print("[ERROR] Error in episode number: ", episode)
                        print(str(e))
                    break
            print("\n\n")

        if len(log_data["id"]) != 0:
            print(
                f"~~~~~~~~~~ Finished {task_type}_{run_type}_{str(args.start_idx)}_{str(args.end_idx)} ~~~~~~~~~~"
            )
            os.makedirs(args.results_dir, exist_ok=True)
            tag = f"_{args.tag}" if args.tag else ""
            result_stem = (
                f"{args.results_dir}/{questioner.__class__.__name__}"
                f"_{args.prompt_variant}_{args.policy}{tag}"
                f"_{task_type}_{run_type}_{args.start_idx}_{args.end_idx}"
            )
            with open(f"{result_stem}.json", "w", encoding="utf-8") as file:
                json.dump(log_data, file, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved {result_stem}.json")

        env.close()
