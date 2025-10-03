"""
Prompt dataset loading and management.
"""

import json
import logging
from pathlib import Path
from typing import Any

import requests
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def load_databricks_dolly(
    tokenizer: PreTrainedTokenizerBase,
    min_input_length: int = 0,
    max_input_length: int = 500,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load Databricks Dolly 15k dataset for prompts.

    Args:
        tokenizer: Tokenizer for counting input tokens
        min_input_length: Minimum prompt length in tokens
        max_input_length: Maximum prompt length in tokens
        cache_dir: Directory to cache dataset (default: current directory)

    Returns:
        List of prompt dictionaries with 'prompt' and 'num_input_tokens' keys
    """
    cache_file = (cache_dir or Path.cwd()) / "databricks-dolly-15k.jsonl"

    # Load from cache if available
    if cache_file.exists():
        logger.info(f"Loading cached dataset from {cache_file}")
        with open(cache_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    else:
        logger.info("Downloading Databricks Dolly 15k dataset")
        url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content

            # Cache the dataset
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                f.write(content)

            dataset = [json.loads(line) for line in content.decode().split("\n") if line.strip()]
            logger.info(f"Dataset downloaded and cached to {cache_file}")

        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            raise RuntimeError(f"Could not download Databricks Dolly dataset: {e}") from e

    # Process prompts and count tokens
    prompts: list[dict[str, Any]] = []
    for item in dataset:
        user_prompt = item.get("context", "") + item.get("instruction", "")
        if not user_prompt:
            continue

        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            num_tokens = len(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize prompt: {e}")
            continue

        if min_input_length <= num_tokens <= max_input_length:
            prompts.append({
                "prompt": user_prompt,
                "num_input_tokens": num_tokens,
            })

    logger.info(
        f"Loaded {len(prompts)} prompts "
        f"(filtered from {len(dataset)} with {min_input_length}-{max_input_length} tokens)"
    )

    if not prompts:
        raise ValueError(
            f"No prompts found within token range {min_input_length}-{max_input_length}"
        )

    return prompts


def load_custom_prompts(
    tokenizer: PreTrainedTokenizerBase,
    prompts_file: Path,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load custom prompts from a JSON/JSONL file.

    Expected format (JSONL):
        {"prompt": "your prompt text"}
        {"prompt": "another prompt"}

    Or JSON array:
        [{"prompt": "your prompt"}, {"prompt": "another prompt"}]

    Args:
        tokenizer: Tokenizer for counting input tokens
        prompts_file: Path to prompts file
        system_prompt: Optional system prompt to prepend

    Returns:
        List of prompt dictionaries with 'prompt' and 'num_input_tokens' keys
    """
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    content = prompts_file.read_text()

    # Try parsing as JSONL first
    try:
        if "\n" in content:
            dataset = [json.loads(line) for line in content.split("\n") if line.strip()]
        else:
            dataset = json.loads(content)
            if not isinstance(dataset, list):
                dataset = [dataset]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON/JSONL file: {e}") from e

    prompts: list[dict[str, Any]] = []
    for item in dataset:
        if isinstance(item, str):
            prompt_text = item
        elif isinstance(item, dict):
            prompt_text = item.get("prompt") or item.get("text") or item.get("content", "")
        else:
            logger.warning(f"Skipping invalid prompt item: {item}")
            continue

        if not prompt_text:
            continue

        # Build chat for tokenization
        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.append({"role": "user", "content": prompt_text})

        try:
            num_tokens = len(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize prompt: {e}")
            continue

        prompts.append({
            "prompt": prompt_text,
            "num_input_tokens": num_tokens,
        })

    logger.info(f"Loaded {len(prompts)} custom prompts from {prompts_file}")

    if not prompts:
        raise ValueError(f"No valid prompts found in {prompts_file}")

    return prompts


def create_single_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Create a single-prompt dataset for testing.

    Args:
        tokenizer: Tokenizer for counting input tokens
        prompt_text: The prompt text
        system_prompt: Optional system prompt

    Returns:
        Single-item list with prompt dictionary
    """
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": prompt_text})

    num_tokens = len(
        tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    return [{
        "prompt": prompt_text,
        "num_input_tokens": num_tokens,
    }]
