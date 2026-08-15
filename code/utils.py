"""
Author: e-zorzi
License: Apache 2.0
"""

from google import genai
import os
import json
import openai
from datetime import datetime
from abc import ABC, abstractmethod
import base64
from io import BytesIO
from attrs import define, field
from typing import Optional, Union, Iterable
import numpy as np
from PIL import Image
from colorama import Fore, init as colorama_init
from uuid import uuid4

colorama_init(autoreset=True)

# Per request. The maximum daily spend (in terms of input tokens) will be
# ((N_tokens / 1_000_000) * RPD * price_per_1M) e.g. for Gemini2.5-pro,
#  which costs around 2$ per 1M tokens (Nov 2025), with this the max daily
# cost will be (20_000/1_000_000) * 10_000 * 2 = $400
_SAFEGUARD_N_TOKENS = 20_000

# Using the very-handwavy 4 letters = 1 token
_SAFEGUARD_N_LETTERS = _SAFEGUARD_N_TOKENS * 4

# For images
_SAFEGUARD_IMAGE_RESOLUTION = 1024


def load_api_keys(dotenv_path: str = None):
    from dotenv import load_dotenv

    if dotenv_path is not None:
        print(f"Loaded dotenv file at {dotenv_path}: {load_dotenv(dotenv_path)}")
    else:
        print(
            f"Loaded dotenv file at $HOME/.env.ml: {load_dotenv(os.path.join(os.environ['HOME'], '.env.ml'))}"
        )


def _warn_requires_vllm(classname, model_id):
    print(
        Fore.YELLOW
        + f"[WARN] `{classname}` requires a connection with a local VLLM server. \
Make sure to run the command `vllm serve {model_id} <options>` in a terminal, and wait for its initialization."
        + Fore.WHITE
    )


def _warn_prompt_too_long(len_prompt, safeguard_length):
    print(
        Fore.YELLOW
        + f"[WARN] The passed prompt has length {len_prompt}, greater than the maximum allowed: {safeguard_length}. It will be truncated accordingly.\
If you want to increase this limit, change the constant _SAFEGUARD_N_LETTERS in the file from which you import this class."
        + Fore.WHITE
    )


def _warn_missing_key(key_name):
    print(Fore.RED + f"[ERROR] {key_name} is missing in the environment." + Fore.WHITE)


def encode_image_b64(image, format):
    im_file = BytesIO()
    image.save(im_file, format=format.upper())
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode("utf-8")


def get_batch_result(
    info_file_path: Union[str, os.PathLike],
) -> tuple[bool, Optional[str]]:
    load_api_keys()
    with open(info_file_path, "r") as read_handle:
        batch_name = read_handle.readlines()[0]
        batch_name.rstrip("\n ")
        assert batch_name.startswith("batches"), "Wrong file passed ?!"

    client = genai.Client()
    batch_job = client.batches.get(name=batch_name)  # Initial get

    # while batch_job.state.name not in completed_states:
    #     print(f"Current state: {batch_job.state.name}")
    #     time.sleep(30)  # Wait for 30 seconds before polling again
    #     batch_job = client.batches.get(name=job_name)

    # print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == "JOB_STATE_FAILED":
        print(Fore.RED + f"[ERROR] {batch_job.error}" + Fore.WHITE)
        return (False, None)
    elif batch_job.state.name == "JOB_STATE_SUCCEEDED":
        result_file_name = batch_job.dest.file_name
        # Create new file name:
        # 1) Find last '.' by reverting the string
        last_dot_index = info_file_path[::-1].index(".")
        # 2) Compute the correct index
        last_dot_index = len(info_file_path) - last_dot_index - 1
        # 3) This is the path without the extension after "."
        path_no_extension = info_file_path[:last_dot_index]
        # 4) This is the final path
        results_file_path = f"{path_no_extension}.results.jsonl"
        print(
            Fore.GREEN
            + f"[SUCCESS] Downloading result file content to {results_file_path} ..."
            + Fore.WHITE
        )
        file_content = client.files.download(file=result_file_name)
        # Process file_content (bytes) as needed
        with open(results_file_path, "a") as write_handle:
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if line != "":
                    write_handle.write(json.dumps(json.loads(line)))
                    write_handle.write("\n")
        return (True, results_file_path)
    elif batch_job.state.name == "JOB_STATE_PENDING":
        print(Fore.YELLOW + "[INFO] Job still pending" + Fore.WHITE)
        print(batch_job)
        return (False, None)
    else:
        print(batch_job)
        return (False, None)


class IRemoteLLM(ABC):
    @abstractmethod
    def ask(
        self,
        *,
        prompt: str,
        images: Iterable[Union[np.ndarray, Image.Image]],
        **kwargs,
    ) -> str:
        pass


@define(kw_only=True, auto_attribs=True)
class GeminiLLM(IRemoteLLM):
    model_id: str
    api_key: str = field(default=None, repr=lambda _: "<|CENSORED|>")
    _delay: float = field(default=0.1)
    include_thoughts: bool = field(default=False)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    aspect_ratio: str = field(default="1:1")
    image_size: str = field(default="1k")

    @aspect_ratio.validator
    def _aspect_ratio_check(self, attr, val):
        if val not in ["1:1", "16:9", "4:3", "3:4", "9:16", "2:3", "3:2"]:
            raise ValueError()

    @image_size.validator
    def _image_size_check(self, attr, val):
        if val not in ["1k", "2k", "4k"]:
            raise ValueError()

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv(
                "GEMINI_API_KEY",
            )

        self._client = genai.Client(api_key=self.api_key)

    def _get_config(
        self,
        thinking_budget=None,
        aspect_ratio=None,
        image_size=None,
        include_thoughts=None,
        generate_images: bool = False,
        as_dict: bool = False,
    ):
        if include_thoughts is None:
            _include_thoughts = include_thoughts
        else:
            _include_thoughts = self.include_thoughts
        if thinking_budget is None:
            thinking_config = (
                genai.types.ThinkingConfig(include_thoughts=_include_thoughts)
                if not as_dict
                else dict(include_thoughts=_include_thoughts)
            )
        else:
            thinking_config = (
                genai.types.ThinkingConfig(
                    include_thoughts=_include_thoughts,
                    thinking_budget=thinking_budget,
                )
                if not as_dict
                else dict(
                    include_thoughts=_include_thoughts,
                    thinking_budget=thinking_budget,
                )
            )

        _TYPE = genai.types.GenerateContentConfig if not as_dict else dict

        if generate_images:
            # Generate image config
            aspect_ratio = (
                aspect_ratio if aspect_ratio is not None else self.aspect_ratio
            )
            self._aspect_ratio_check("aspect_ratio", aspect_ratio)
            image_size = image_size if image_size is not None else self.image_size
            self._image_size_check("image_size", image_size)

            image_config = (
                genai.types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size=image_size
                )
                if not as_dict
                else dict(aspect_ratio=aspect_ratio, image_size=image_size)
            )
            if generate_images:
                response_modalities = ["TEXT", "IMAGE"]
            else:
                response_modalities = ["TEXT"]

            return _TYPE(
                temperature=self.temperature,
                top_p=self.top_p,
                thinking_config=thinking_config,
                image_config=image_config,
                response_modalities=response_modalities,
            )
        else:
            return _TYPE(
                temperature=self.temperature,
                top_p=self.top_p,
                thinking_config=thinking_config,
            )

    def _image_text_chat(
        self,
        prompt,
        image,
        thinking_budget=None,
        return_metadata: bool = False,
    ):
        # Handle arrays
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_format = "png"
        else:
            image_format = image.format

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        # Safety checks
        height, width = image.size
        if height > _SAFEGUARD_IMAGE_RESOLUTION or width > _SAFEGUARD_IMAGE_RESOLUTION:
            raise ValueError(
                f"Image size safeguard: passed an image of resolution {width} x {height}, larger than the safeguard {_SAFEGUARD_IMAGE_RESOLUTION} x {_SAFEGUARD_IMAGE_RESOLUTION}"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        image_bytes = BytesIO()
        image.save(image_bytes, format=image_format.upper())
        image_bytes = image_bytes.getvalue()

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=f"image/{image_format.lower()}",
                ),
                prompt[:_SAFEGUARD_N_LETTERS],
            ],
            config=self._get_config(thinking_budget),
        )

        if return_metadata:
            return (response.text, response.usage_metadata)
        else:
            return response.text

    def _text_chat(
        self,
        prompt,
        thinking_budget=None,
        return_metadata: bool = False,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[prompt[:_SAFEGUARD_N_LETTERS]],
            config=self._get_config(thinking_budget),
        )
        if return_metadata:
            return (response.text, response.usage_metadata)
        else:
            return response.text

    def ask(
        self,
        *,
        prompt: str,
        images: Iterable[Union[np.ndarray, Image.Image]] = None,
        thinking_budget=None,
        return_metadata: bool = False,
        **kwargs,
    ) -> str:
        """Primary method for generating (multimodal or unimodal) requests

        Args:
            prompt (str): text prompt
            images (Iterable[Union[np.ndarray, &quot;Image&quot;]], optional): a set of images related to the prompt, if a multimodal chat is required

        Returns:
            str: the response of the model
        """
        if images is not None:
            assert len(images) == 1, "Only 1 image is supported at the moment"
            return self._image_text_chat(
                prompt,
                images[0],
                thinking_budget=thinking_budget,
                return_metadata=return_metadata,
                **kwargs,
            )
        else:
            return self._text_chat(
                prompt,
                thinking_budget=thinking_budget,
                return_metadata=return_metadata,
                **kwargs,
            )


@define(kw_only=True, auto_attribs=True)
class OpenAILLM(IRemoteLLM):
    model_id: str
    api_key: str = field(default=None, repr=lambda _: "<|CENSORED|>")
    _url: str = field(default="https://api.openai.com/v1")
    _delay: float = field(default=0.1)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    max_output_length: int = field(default=12000)

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        try:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)
        except openai.OpenAIError as e:
            _warn_missing_key("OPENAI_API_KEY")
            raise e

    def _build_answer(self, completion, **kwargs):
        stringbuilder = ""
        logprobs = []
        for chunk in completion:
            token = chunk.choices[0].delta.content
            if "logprobs" in kwargs and chunk.choices[0].logprobs is not None:
                logprobs.append(chunk.choices[0].logprobs.content[0].top_logprobs)
            if token:
                stringbuilder += f"{token}"
        if "logprobs" in kwargs:
            return stringbuilder, logprobs
        else:
            return stringbuilder

    def _image_text_chat(self, prompt, image, **kwargs):
        # Handle arrays
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_format = "png"
        else:
            image_format = image.format

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        # Safety checks
        height, width = image.size
        if height > _SAFEGUARD_IMAGE_RESOLUTION or width > _SAFEGUARD_IMAGE_RESOLUTION:
            raise ValueError(
                f"Image size safeguard: passed an image of resolution {width} x {height}, larger than the safeguard {_SAFEGUARD_IMAGE_RESOLUTION} x {_SAFEGUARD_IMAGE_RESOLUTION}"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        image_bytes = encode_image_b64(image, image_format)
        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format.lower()};base64,{image_bytes}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
            **kwargs,
        )

        return self._build_answer(completion, **kwargs)

    def _text_chat(
        self,
        prompt,
        **kwargs,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
            **kwargs,
        )

        return self._build_answer(completion, **kwargs)

    def ask(
        self,
        *,
        prompt: str,
        images: Iterable[Union[np.ndarray, Image.Image]] = None,  # type: ignore
        **kwargs,
    ) -> str:
        """_summary_

        Args:
            prompt (str): text prompt
            images (Iterable[Union[np.ndarray, &quot;Image&quot;]], optional): a set of images related to the prompt, if a multimodal chat is required

        Returns:
            str: the response of the model
        """
        if images is not None:
            assert len(images) == 1, "Only 1 image is supported at the moment"
            return self._image_text_chat(
                prompt,
                images[0],
                **kwargs,
            )
        else:
            return self._text_chat(
                prompt,
                **kwargs,
            )


@define(kw_only=True, auto_attribs=True)
class ClientBasedLLM(OpenAILLM):
    """Client that handles a VLLM connection. The model_id passed to __init__ should match the model running via VLLM."""

    api_key: str = field(default="EMPTY")
    _port: int = field(default=8000)
    _url: Optional[str] = None

    def __attrs_post_init__(self):
        _warn_requires_vllm(self.__class__.__name__, self.model_id)

        if self._url is None:
            self._url = f"http://localhost:{self._port}/v1"

        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)
