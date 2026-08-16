from abc import ABC, abstractmethod
import re
import numpy as np
from utils import ClientBasedLLM
from retrying import retry
import time


_SCORE_RE = re.compile(r"<score>\s*([012])\s*</score>", re.IGNORECASE | re.DOTALL)
_MOTIVATION_RE = re.compile(
    r"<motivation>(.*?)</motivation>", re.IGNORECASE | re.DOTALL
)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.IGNORECASE | re.DOTALL)
# Fallback parsers for slightly off-format outputs.
_SCORE_FALLBACK_RE = re.compile(
    r"(?:<score>|score\s*[:=]\s*)([012])(?:\s*</score>)?",
    re.IGNORECASE,
)
_GENERIC_FORCE_ASK_DESC_TYPES = {
    "category",
    "color",
    "context",
}


def _parse_questioner_response(response: str):
    """Parse the official <motivation>/<score>/<question> format."""
    motivation_match = _MOTIVATION_RE.search(response)
    score_match = _SCORE_RE.search(response)
    question_match = _QUESTION_RE.search(response)

    if score_match is None:
        score_match = _SCORE_FALLBACK_RE.search(response)
    if score_match is None:
        raise ValueError(f"Could not parse <score> from response: {response!r}")

    score = int(score_match.group(1))
    reasoning = (
        motivation_match.group(1).strip()
        if motivation_match is not None
        else response.strip()[:200]
    )
    question = question_match.group(1).strip() if question_match is not None else ""
    if question in ("", "''", '""', "None", "none"):
        question = None

    if score == 1:
        if not question:
            raise ValueError(
                f"Score is 1 but no question was provided. Response: {response!r}"
            )
        return dict(question=question, conclusion=None, reasoning=reasoning, score=score)
    if score == 2:
        return dict(question=None, conclusion=1, reasoning=reasoning, score=score)
    return dict(question=None, conclusion=0, reasoning=reasoning, score=score)


def _default_discriminative_question(description: str) -> str:
    """Fallback clarifying question when the model must ask but gave none."""
    desc = (description or "").strip()
    if not desc:
        return "What is the most distinctive visual feature of the target object?"
    # Prefer asking about attributes not already spelled out.
    return (
        f"Besides what is already stated in '{desc}', "
        "what other distinctive visual details uniquely identify the target object?"
    )


def _is_short_description(description: str, desc_type: str | None = None) -> bool:
    if desc_type in _GENERIC_FORCE_ASK_DESC_TYPES:
        return True
    words = [w for w in (description or "").split() if w]
    return len(words) <= 4


# Example of a questioner prompt (challenge repo default; kept for comparison)
QUESTIONER_EXAMPLE_PROMPT = (
    "An oracle has a fixed target image and has given you this description of the object in the target image: {TARGET_DESCRIPTION}. "
    "You are given the above image, which may or may not picturing the same object as the target image owned by the oracle. Your goal is to decide whether the "
    "image that you see corresponds to the same image target image owned by the oracle. "
    "Use, to guide your decision, the description of the object, the give image, and, if they exists, the questions that you previously asked to the oracle "
    "and the associated answers. Provide a reasoning about your conclusion, or why you are uncertain and asked a question. "
    "If you are sure that the two image match, return the score 2, if you are sure that they don't match, return "
    "the score 0. If you are unsure either way, return the score 1 and ask an informative question to the oracle about what it might appears in the target image, "
    "to dispel your doubts. You can always trust the oracle's answers and the initial description. Do not ask questions that can be directly answered by "
    "reading the initial description, or questions about the image that you are provided. Be careful: the target image and the given image might differ only in some small details, "
    "like the color of the object, its texture, or the presence of other objects. For example, the target image might picture a bed with a blue comforter, "
    "and the given image a bed with a red comforter, or a blue bed but with a white comforter. "
    "The image might have distortions or digital artifacts: *NEVER* mention them in the question. Prefer asking question if the description is very generic. "
    "Strictly follow this output format: "
    "<motivation>Your reasoning here (under 60 words, do NOT use double quotes \")</motivation><score>0, 1, or 2</score><question>Your question or '' (if score is not 1)</question>"
)

# Light-CoNav prompt from arXiv:2604.00265 (Appendix, Prompts).
# USER_TASK is substituted with description D; image is prepended separately by the VLM client.
QUESTIONER_PAPER_PROMPT = (
    "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. "
    "You were given the following task by the user '{USER_TASK}'.\n"
    "Currently, you are facing a scene represented by the given image. Reason about what you are seeing, comparing what you know "
    "about the task (given the user commands) and the given scene.\n"
    "For example, if the description is 'Black leather sofa near a lampstand' your reasoning process will be "
    "'I'm currently observing a brown sofa which is different than black, making it unlikely to be the target sofa. "
    "Moreover, there is no lampstand near it, only a rug and a window' etc.\n"
    "If there are distortions or artifacts, do not focus on them, focus on the object at hand. At the end of the reasoning process, "
    "evaluate how well the provided image aligns with the user's task.\n"
    "Assign a confidence score based on the following scale:\n"
    "- 0: You are certain the image **DOES NOT** match the task.\n"
    "- 1: You are unsure whether the image matches the task or not.\n"
    "- 2: You are certain the image **DOES** match the task.\n"
    "Provide a concise reasoning (under 100 words). If you are unsure (score 1), ask an informative question to the user: "
    "this question should help you dispel any doubt about the current observation. Otherwise, just return ''. "
    "Use your reasoning output to ask the best question possible. Prefer asking questions at the start, rather than blindly "
    "coming to a conclusion, if the description is very short and does not include many details. "
    "The image might have distortions or digital artifacts: *NEVER* mention them in the question. "
    "Consider the information contained in the previous questions and answers, which are provided here.\n"
    "{CONTEXT}\n"
    "Strictly follow this output format: "
    "<motivation>Your reasoning here</motivation><score>0, 1, or 2</score><question>Your question or None (if score is not 1)</question>"
)

# paper prompt + ProCompNav-style yes/no discriminative-question constraint.
# Kept as a separate variant so 'paper' baselines stay comparable.
QUESTIONER_PAPER_YN_PROMPT = QUESTIONER_PAPER_PROMPT.replace(
    "ask an informative question to the user: "
    "this question should help you dispel any doubt about the current observation.",
    "ask an informative yes/no question to the user: "
    "the question must be answerable with yes or no, must target a visible "
    "attribute of the object (color, material, nearby objects, position) that is "
    "NOT already stated in the task description, and must help you confirm or "
    "exclude the match in a single step.",
)

# Adaptive prompt: encodes "ask only when the description lacks information"
# (TANDEM-style cost-aware + ProCompNav discriminative questions) directly in
# the prompt, so the model decides per-description whether to ask or conclude.
# This avoids programmatic per-type prompt switching and works on unseen
# descriptions at test time.
QUESTIONER_PAPER_ADAPTIVE_PROMPT = QUESTIONER_PAPER_YN_PROMPT.replace(
    "Use your reasoning output to ask the best question possible. Prefer asking questions at the start, rather than blindly "
    "coming to a conclusion, if the description is very short and does not include many details. ",
    "Use your reasoning output to ask the best question possible. "
    "Decide whether to ask based on how informative the task description is. "
    "If the description is generic, short, or lacks specific visual details "
    "(e.g. just a category name like 'Chair'), the image alone is usually not "
    "enough to tell the target from similar distractors: you SHOULD ask one "
    "discriminative yes/no question. "
    "If the description already provides sufficient details (e.g. color, feature, "
    "and context), reason directly from the image and conclude without asking: "
    "do not ask a question when you already have enough information to decide. ",
)

# Our_Prompt_V1: 我们在 E2 之后讨论设计的第一个自研 prompt（2026-08-02）。
# 核心设计点（详见 PROMPT_DESIGN_NOTES.md）：
#  1. 决策充分性 = 描述信息量 + 当前图可见证据 + 历史答案，三者联合而非描述信息量单方面决定
#     （修正 E2 adaptive "信息充足就不能问" 的错）
#  2. 显式 OBSERVE→COMPARE→DECIDE 三步 CoT（对应论文"先推理再打分"）
#  3. score 1 时问题必须从目标图获取新信息（官方 example 的 orale 设定），
#     而非针对当前图（paper_yn/adaptive 的偏离），以便积累跨图可用的目标画像
#  4. 采纳官方 example 的 orale 角色设定、决策信息源（描述+当前图+历史问答）、
#     信任 orale 与初始描述、禁止问描述已答或当前图内容、微小差异提醒
#  5. 显式 "Reason first, score second"
# {USER_TASK} 与 {CONTEXT} 在 _build_prompt 中填充。
OUR_PROMPT_V1 = (
    "An oracle holds a fixed target image and gave you this description of the object in it: '{USER_TASK}'.\n"
    "You are given a candidate image, which may or may not show the same object as the target image. "
    "Decide whether the candidate shows the target object, using: the description, the candidate image, "
    "and any previous questions you asked the oracle with its answers.\n"
    "\n"
    "Reason step by step, then give the score. Write your reasoning in <motivation> in exactly these three steps:\n"
    "Step 1 - OBSERVE the candidate image. List the concrete visual facts relevant to the description: "
    "the object, its attributes (color, material, features), and nearby objects. Be precise.\n"
    "Step 2 - COMPARE each attribute of the description against what you observed. "
    "For each, say whether it is MATCHED, CONTRADICTED, or UNVERIFIABLE in the candidate image "
    "(occluded, out of view, too small to tell). Also incorporate the previous questions and answers:\n"
    "{CONTEXT}\n"
    "Step 3 - DECIDE. Give score 2 if every relevant attribute is MATCHED and nothing relevant is UNVERIFIABLE. "
    "Give score 0 if any attribute is CONTRADICTED. Give score 1 otherwise (some attribute is UNVERIFIABLE, "
    "or the description is too vague to check).\n"
    "\n"
    "Reason first, score second: never give a score without the reasoning in <motivation> that supports it.\n"
    "\n"
    "If you give score 1, ask exactly ONE question that obtains from the oracle NEW information about the "
    "TARGET object - never about the candidate image. The question must:\n"
    "  - target an attribute NOT already stated in the description or in earlier answers "
    "(e.g. color, material, nearby objects, position);\n"
    "  - be answerable from the target image;\n"
    "  - also be checkable in the candidate image, so the answer settles this decision and remains useful "
    "for future candidates.\n"
    "The answers you collect build a profile of the target object that stays valid across images: use it, "
    "and do not re-ask what is already answered.\n"
    "Trust the oracle's answers and the initial description. Do not ask what the description already states. "
    "Do not ask about the candidate image you are looking at.\n"
    "Remember the target and candidate images may differ only in small details: the color, the texture, "
    "or the presence of other objects. The image may have distortions or digital artifacts: "
    "NEVER mention them in the question.\n"
    "\n"
    "Strictly follow this output format: "
    "<motivation>Step 1: ... Step 2: ... Step 3: ...</motivation><score>0, 1, or 2</score>"
    "<question>Your question or None (if score is not 1)</question>"
)

OUR_PROMPT_V2 = (
    "An oracle holds a fixed target image and gave you this description of the object in it: '{USER_TASK}'.\n"
    "You are given a candidate image, which may or may not show the same object as the target image. "
    "Decide whether the candidate shows the target object, using: the description, the candidate image, "
    "and any previous questions you asked the oracle with its answers.\n"
    "\n"
    "Reason step by step, then give the score. Write your reasoning in <motivation> in exactly these three steps. "
    "Keep each step to 1-2 short sentences; the whole reasoning must stay under 120 words.\n"
    "Step 1 - OBSERVE the candidate image. State the main object and only the visual facts relevant to the "
    "description and previous answers: type, color, material, salient features, nearby objects.\n"
    "Step 2 - COMPARE each attribute from the description AND from the previous answers against what you observed. "
    "For each, say MATCHED, CONTRADICTED, or UNVERIFIABLE (occluded, out of view, too small to tell). "
    "Previous questions and answers:\n"
    "{CONTEXT}\n"
    "Step 3 - DECIDE by these rules:\n"
    "- Score 0 if any attribute from the description or the previous answers is CONTRADICTED, or if the "
    "candidate contains no object of the described type at all.\n"
    "- Score 2 only if the description and the previous answers together provide at least one distinguishing "
    "attribute beyond the object type (color, material, salient feature, position, or nearby objects), and "
    "every checkable attribute is MATCHED with nothing relevant UNVERIFIABLE.\n"
    "- Score 1 otherwise. In particular, if the description only names an object type and no previous answer "
    "pins the target down, an object of the right type is NOT enough evidence - other images may contain the "
    "same kind of object - so score 1 and ask. Also score 1 when a stated attribute is UNVERIFIABLE.\n"
    "\n"
    "Reason first, score second: never give a score without the reasoning in <motivation> that supports it.\n"
    "\n"
    "If you give score 1, ask exactly ONE question that obtains from the oracle NEW information about the "
    "TARGET object - never about the candidate image. The question must:\n"
    "- target an attribute NOT already stated in the description or in earlier answers "
    "(e.g. color, material, salient features, nearby objects, position);\n"
    "- be answerable from the target image;\n"
    "- also be checkable in the candidate image, so the answer settles this decision and stays useful for "
    "future candidates.\n"
    "If the description only names an object type, ask for the attribute that best distinguishes the target "
    "from other objects of the same type.\n"
    "The answers you collect build a profile of the target object that stays valid across images: use it, "
    "and do not re-ask what is already answered.\n"
    "Trust the oracle's answers and the initial description. Do not ask what the description already states. "
    "Do not ask about the candidate image you are looking at.\n"
    "Remember the target and candidate images may differ only in small details: the color, the texture, "
    "or the presence of other objects. The image may have distortions or digital artifacts: "
    "NEVER mention them in the question.\n"
    "\n"
    "Strictly follow this output format: "
    "<motivation>Step 1: ... Step 2: ... Step 3: ...</motivation><score>0, 1, or 2</score>"
    "<question>Your question or None (if score is not 1)</question>\n"
)

OUR_PROMPT_V3 = (
    "An oracle holds a fixed target image and gave you this description of the object in it: '{USER_TASK}'.\n"
    "You are given a candidate image, which may or may not show the same object as the target image. "
    "Decide whether the candidate shows the target object, using only: the description, the candidate image, "
    "and the previous questions you asked the oracle with its answers.\n"
    "\n"
    "Reason step by step, then give the score. Write your reasoning in <motivation> in exactly these three steps. "
    "Keep each step to 1-2 short sentences; keep the complete reasoning under 120 words.\n"
    "Step 1 - OBSERVE the candidate image. State the main object of the described type and only the visual facts "
    "relevant to the description and previous answers: type, color, material, salient features, nearby objects. "
    "If no object of the described type is present, say so.\n"
    "Step 2 - COMPARE. Check ONLY the attributes explicitly stated in the description or in the previous answers; "
    "do not invent extra attributes to check. For each stated attribute, say MATCHED, CONTRADICTED, or UNVERIFIABLE "
    "(occluded, cropped out, or too small to tell). Judge colors and materials with tolerance for lighting and "
    "viewing angle: mark CONTRADICTED only when the difference is clear. A stated nearby object is CONTRADICTED "
    "only if the surrounding area is clearly visible and the object is absent; if the area is out of view, it is "
    "UNVERIFIABLE. Do not mark attributes that were never stated as UNVERIFIABLE. Previous questions and answers:\n"
    "{CONTEXT}\n"
    "Step 3 - DECIDE by these rules, in order:\n"
    "- Score 0 if the candidate contains no object of the described type, or if any stated attribute is clearly "
    "CONTRADICTED.\n"
    "- Score 1 if the description only names the object type and no previous answer adds a distinguishing "
    "attribute: an object of the right type alone is NOT enough evidence, so ask. Also score 1 if any stated "
    "attribute is UNVERIFIABLE.\n"
    "- Score 2 if every stated attribute is MATCHED and none is CONTRADICTED or UNVERIFIABLE. When the description "
    "states at least one attribute beyond the object type and everything checks out, commit to score 2; do not ask "
    "extra questions just to be safe. Before outputting score 2, verify that Step 2 contains no UNVERIFIABLE result "
    "for any stated attribute; if it does, output score 1 instead. Never write 'matched or unverifiable' and score 2.\n"
    "\n"
    "Reason first, score second: never give a score without the reasoning in <motivation> that supports it.\n"
    "\n"
    "If you give score 1, ask exactly ONE short question that obtains from the oracle NEW information about the "
    "TARGET object - never about the candidate image. The question must:\n"
    "- ask about exactly one attribute (color, material, salient feature, position, or one nearby object); never "
    "combine two attributes in one question;\n"
    "- target an attribute NOT already stated in the description or answered before;\n"
    "- be answerable by looking at the target image, and checkable in candidate images, so the answer settles this "
    "decision and stays useful for future candidates;\n"
    "- pick the attribute that best distinguishes the target from other objects of the same type;\n"
    "- be neutral: do not presume that something you see in the current candidate is also true of the target.\n"
    "The answers you collect build a profile of the target object that stays valid across images: use it, and do "
    "not re-ask what is already answered.\n"
    "Trust the oracle's answers and the initial description. Do not ask what the description already states. "
    "Do not ask about the candidate image you are looking at.\n"
    "Remember the target and candidate images may differ only in small details: the color, the texture, or the "
    "presence of other objects. The image may have distortions or digital artifacts: NEVER mention them in the "
    "question or reasoning.\n"
    "\n"
    "Strictly follow this output format: "
    "<motivation>Step 1: ... Step 2: ... Step 3: ...</motivation><score>0, 1, or 2</score>"
    "<question>Your question or None (if score is not 1)</question>\n"
)

PROMPT_VARIANTS = {
    "example": QUESTIONER_EXAMPLE_PROMPT,
    "paper": QUESTIONER_PAPER_PROMPT,
    "paper_yn": QUESTIONER_PAPER_YN_PROMPT,
    "paper_adaptive": QUESTIONER_PAPER_ADAPTIVE_PROMPT,
    "our_prompt_v1": OUR_PROMPT_V1,
    "our_prompt_v2": OUR_PROMPT_V2,
    "our_prompt_v3": OUR_PROMPT_V3,
}

# Wording signals used by the uncertainty-gate policy as a stand-in for
# output-logit entropy (vLLM HTTP API does not expose token logprobs here).
_UNCERTAIN_PHRASES = (
    "not sure",
    "not certain",
    "uncertain",
    "unsure",
    "cannot tell",
    "can't tell",
    "hard to tell",
    "ambiguous",
    "unclear",
    "maybe",
    "perhaps",
    "possibly",
    "could be",
    "might be",
    "seems",
    "likely",
    "probably",
)


def _looks_uncertain(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in _UNCERTAIN_PHRASES)


def _validate_observation(observation):
    assert isinstance(observation["image"], np.ndarray) and (
        not observation["answer"] or isinstance(observation["answer"], str)
    ), (
        "Wrong observation format: it must be a dictionary with keys 'image' and 'answer', where 'answer' is a numpy array and 'answer' is either a string or None"
    )
    assert (
        len(observation["image"].shape) == 3 and observation["image"].shape[2] == 3
    ), "Wrong image format: must be a numpy array of shape (H,W,3) --- an rgb image."


class QuestionerInterface(ABC):
    """Abstract Questioner class. Your questioner should inherit from this."""

    def __init__(self, info, *args):
        self.info = info  # required info like the task description
        self.target_description = info["target_description"]

    @abstractmethod
    def ask_or_conclude(self, observation):
        # TODO: this is what you have to implement
        pass

    def add_answer(self, answer):
        self.answers.append(answer)

    def reset_questions(self):
        self.questions = []
        self.answers = []

    def reset_time(self):
        self.time_required = 0


class QuestionerLocalVLM(QuestionerInterface):
    """Local VLM questioner with optional anti-FP decision policies."""

    def __init__(
        self,
        info,
        model_id: str,
        port: int = 8001,
        prompt_variant: str = "our_prompt_v3",
        temperature: float = 0.0,
        policy: str = "dedup_category_only",
        description_type: str | None = None,
        max_questions_per_obs: int = 2,
    ):
        # info will contain the target object description: info["target_description"]
        # This is also saved in self.target_description
        super().__init__(info)
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant={prompt_variant!r}. "
                f"Choose one of {sorted(PROMPT_VARIANTS)}"
            )
        if policy not in {
            "baseline",
            "anti_fp",
            "force_ask_short",
            "uncertainty_gate",
            "dedup_force_decide",
            "dedup_category_only",
        }:
            raise ValueError(
                f"Unknown policy={policy!r}. "
                "Choose one of baseline|anti_fp|force_ask_short|"
                "uncertainty_gate|dedup_force_decide|dedup_category_only"
            )
        self.prompt_variant = prompt_variant
        self.policy = policy
        self.description_type = description_type
        self.max_questions_per_obs = max_questions_per_obs
        self.client = ClientBasedLLM(
            model_id=model_id,
            port=port,
            url=f"http://localhost:{port}/v1",
            temperature=temperature,
            top_p=1.0,
            max_output_length=2048,
        )
        self.questions = []
        self.reasonings = []
        self.answers = []
        self.time_required = 0
        self.n_questions = 0
        self._questions_on_current_obs = 0

    def _build_prompt(self, observation):
        # A new oracle answer arrives only after we previously asked a question.
        # Guard against duplicates when @retry re-enters this method.
        if observation.get("answer") and len(self.answers) < len(self.questions):
            self.answers.append(observation["answer"])
            # An answer means we are still on the same observation.
            pass

        if self.prompt_variant == "paper":
            if self.questions and self.answers:
                history_lines = []
                for i, (q, a) in enumerate(
                    zip(self.questions, self.answers), start=1
                ):
                    history_lines.append(
                        f"{i}. {q} <|answer|>{a}<|answer|>"
                    )
                context = "\n".join(history_lines)
            else:
                context = "There are no previous questions or answers."
            return QUESTIONER_PAPER_PROMPT.format(
                USER_TASK=self.target_description,
                CONTEXT=context,
            )

        # Our_Prompt_V1: same historical-Q&A context format as 'paper'.
        if self.prompt_variant == "our_prompt_v1":
            if self.questions and self.answers:
                history_lines = []
                for i, (q, a) in enumerate(
                    zip(self.questions, self.answers), start=1
                ):
                    history_lines.append(
                        f"{i}. {q} <|answer|>{a}<|answer|>"
                    )
                context = "\n".join(history_lines)
            else:
                context = "There are no previous questions or answers."
            return OUR_PROMPT_V1.format(
                USER_TASK=self.target_description,
                CONTEXT=context,
            )

        if self.prompt_variant == "our_prompt_v2":
            if self.questions and self.answers:
                history_lines = []
                for i, (q, a) in enumerate(
                    zip(self.questions, self.answers), start=1
                ):
                    history_lines.append(
                        f"{i}. {q} <|answer|>{a}<|answer|>"
                    )
                context = "\n".join(history_lines)
            else:
                context = "There are no previous questions or answers."
            return OUR_PROMPT_V2.format(
                USER_TASK=self.target_description,
                CONTEXT=context,
            )

        if self.prompt_variant == "our_prompt_v3":
            if self.questions and self.answers:
                history_lines = []
                for i, (q, a) in enumerate(
                    zip(self.questions, self.answers), start=1
                ):
                    history_lines.append(
                        f"{i}. {q} <|answer|>{a}<|answer|>"
                    )
                context = "\n".join(history_lines)
            else:
                context = "There are no previous questions or answers."
            return OUR_PROMPT_V3.format(
                USER_TASK=self.target_description,
                CONTEXT=context,
            )

        # Legacy challenge-repo example prompt.
        prompt_to_use = QUESTIONER_EXAMPLE_PROMPT.format(
            TARGET_DESCRIPTION=self.target_description
        )
        if self.questions and self.answers:
            history = []
            for q, a in zip(self.questions, self.answers):
                history.append(f"Q: {q}\nA: {a}")
            prompt_to_use += (
                "\nHere are the previous questions and answers:\n"
                + "\n".join(history)
            )
        return prompt_to_use

    def _apply_policy(self, action, observation):
        """Post-process model action to reduce false-positive matches."""
        short = _is_short_description(
            self.target_description, self.description_type
        )
        asked = len(self.questions)
        q_on_obs = self._questions_on_current_obs
        is_fresh_obs = observation.get("answer") is None and q_on_obs == 0

        if self.policy == "baseline":
            return action

        # Dedup + cap: if the model re-asks an already-asked question, or
        # exceeds the per-observation budget, force it to decide using the
        # answers it already has instead of looping.
        if self.policy in {"dedup_force_decide", "dedup_category_only"}:
            if (
                self.policy == "dedup_category_only"
                and self.description_type != "category"
            ):
                return action
            if action.get("question") is not None and (
                q_on_obs >= self.max_questions_per_obs
                or self._is_duplicate_question(action["question"])
            ):
                return self._force_decide(observation, action)
            return action

        # Confidence gate (AIUTA-style, wording-based stand-in for entropy):
        # if the model claims a firm conclusion (score 0/2) but its reasoning
        # wavers, treat it as actually unsure -> force a question instead.
        if self.policy == "uncertainty_gate" and action.get("conclusion") is not None:
            if _looks_uncertain(action.get("reasoning")):
                q = action.get("question") or _default_discriminative_question(
                    self.target_description
                )
                return dict(
                    question=q,
                    conclusion=None,
                    reasoning=(
                        (action.get("reasoning") or "")
                        + " [policy: uncertain wording -> ask]"
                    ).strip(),
                    score=1,
                )

        # Block premature MATCH on a fresh observation.
        # force_ask_short: only for short/generic descriptions.
        # anti_fp / uncertainty_gate: for all descriptions (FR-first; NQ may rise).
        block_premature_match = False
        if action.get("conclusion") == 1 and is_fresh_obs:
            if self.policy in {"anti_fp", "uncertainty_gate"}:
                block_premature_match = True
            elif self.policy == "force_ask_short" and short:
                block_premature_match = True

        if block_premature_match:
            q = action.get("question") or _default_discriminative_question(
                self.target_description
            )
            return dict(
                question=q,
                conclusion=None,
                reasoning=(
                    (action.get("reasoning") or "")
                    + " [policy: force ask before match on fresh observation]"
                ).strip(),
                score=1,
            )

        # Cap questions per observation → conservative reject.
        if (
            action.get("question") is not None
            and q_on_obs >= self.max_questions_per_obs
        ):
            return dict(
                question=None,
                conclusion=0,
                reasoning=(
                    (action.get("reasoning") or "")
                    + " [policy: max questions reached → reject]"
                ).strip(),
                score=0,
            )

        return action

    _QUESTION_STOPWORDS = frozenset(
        "what is the of a an does do have has are there it its in on with "
        "and or to how many any which type kind".split()
    )

    def _is_duplicate_question(self, question: str) -> bool:
        def toks(q):
            words = re.findall(r"[a-z]{2,}", q.lower())
            return set(words) - self._QUESTION_STOPWORDS

        qt = toks(question)
        for prev in self.questions:
            if question.strip().lower() == prev.strip().lower():
                return True
            pt = toks(prev)
            if qt and pt and len(qt & pt) / len(qt | pt) >= 0.6:
                return True
        return False

    def _force_decide(self, observation, action):
        """Re-query the model demanding a 0/2 conclusion; fallback reject."""
        prompt = self._build_prompt(observation) + (
            "\n\nIMPORTANT UPDATE: You have already used your question budget "
            "for this candidate; asking again is not allowed. Using the image, "
            "the description, and the previous answers above, you MUST decide "
            "now. Re-check each answered attribute against the candidate: if "
            "any answered attribute is clearly contradicted by the candidate, "
            "output score 0; if the answered attributes are consistent with "
            "the candidate and nothing is contradicted, output score 2. Keep "
            "the same output format with <question>None</question>."
        )
        try:
            response = self.client.ask(
                prompt=prompt, images=[observation["image"]]
            )
            forced = _parse_questioner_response(response)
        except Exception:
            forced = None
        if forced and forced.get("conclusion") is not None:
            forced["reasoning"] = (
                (forced.get("reasoning") or "")
                + " [policy: question budget/duplicate -> forced decision]"
            ).strip()
            return forced
        return dict(
            question=None,
            conclusion=0,
            reasoning=(
                (action.get("reasoning") or "")
                + " [policy: duplicate question and no decision -> reject]"
            ).strip(),
            score=0,
        )

    def notify_new_observation(self):
        """Call when the environment advances to a new distractor image."""
        self._questions_on_current_obs = 0

    @retry(
        stop_max_attempt_number=5,
        wait_exponential_multiplier=2000,
        wait_exponential_max=60000,
    )
    def ask_or_conclude(self, observation):
        _validate_observation(observation)
        start_time = time.time()

        # If queues are synced and no oracle answer, this is a new decision
        # round on (possibly) a new image — reset per-obs counter.
        if observation.get("answer") is None and len(self.questions) == len(
            self.answers
        ):
            self._questions_on_current_obs = 0

        prompt_to_use = self._build_prompt(observation)

        response = self.client.ask(
            prompt=prompt_to_use,
            images=[observation["image"]],
        )

        end_time = time.time()
        self.time_required += end_time - start_time

        action = _parse_questioner_response(response)
        action = self._apply_policy(action, observation)
        self.reasonings.append(action["reasoning"])
        if action["question"] is not None:
            self.questions.append(action["question"])
            self.n_questions += 1
            self._questions_on_current_obs += 1
        return action


class YourQuestioner(QuestionerInterface):
    def __init__(self, info, *args):
        super().__init__(info)
        # TODO
        raise NotImplementedError("Implement your Questioner")

    def ask_or_conclude(self, observation):
        _validate_observation(observation)
        # TODO
        ## Return either (if uncertain whether the observation corresponds to the target description or not)
        # return dict(question=question, conclusion=None, reasoning=reasoning)
        ## if certain that is a match
        # return dict(question=None, conclusion=1, reasoning=reasoning)
        ## or if certaint that is NOT a match
        # return dict(question=None, conclusion=0, reasoning=reasoning)
