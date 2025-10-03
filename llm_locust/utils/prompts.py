"""
Prompt dataset loading and management.

Supports multiple popular datasets for different LLM testing scenarios:
- Databricks Dolly 15k (general Q&A)
- ShareGPT (conversational)
- CNN/DailyMail (summarization)
- BillSum (long context prefill)
- Infinity Instruct (long context decode)
- Shared Prefix (prefix caching)
- Custom JSON/JSONL files
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
        cache_dir: Directory to cache dataset (default: datasets/)

    Returns:
        List of prompt dictionaries with 'prompt' and 'num_input_tokens' keys
    """
    default_cache_dir = Path.cwd() / "datasets"
    cache_file = (cache_dir or default_cache_dir) / "databricks-dolly-15k.jsonl"

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


def load_sharegpt(
    tokenizer: PreTrainedTokenizerBase,
    min_input_length: int = 0,
    max_input_length: int = 2048,
    cache_dir: Path | None = None,
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load ShareGPT dataset for conversational prompts.

    ShareGPT contains real user conversations, ideal for testing chat models.

    Args:
        tokenizer: Tokenizer for counting input tokens
        min_input_length: Minimum prompt length in tokens
        max_input_length: Maximum prompt length in tokens
        cache_dir: Directory to cache dataset (default: current directory)
        num_samples: Limit number of samples (None = all)

    Returns:
        List of prompt dictionaries with 'prompt' and 'num_input_tokens' keys
    """
    default_cache_dir = Path.cwd() / "datasets"
    cache_file = (cache_dir or default_cache_dir) / "sharegpt.jsonl"

    # Load from cache if available
    if cache_file.exists():
        logger.info(f"Loading cached ShareGPT dataset from {cache_file}")
        with open(cache_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()]
            if num_samples:
                dataset = dataset[:num_samples]
    else:
        logger.info("Downloading ShareGPT dataset (this may take a while...)")
        url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            dataset = response.json()

            # Cache as JSONL
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")

            logger.info(f"Dataset downloaded and cached to {cache_file}")

            if num_samples:
                dataset = dataset[:num_samples]

        except requests.RequestException as e:
            logger.error(f"Failed to download ShareGPT dataset: {e}")
            raise RuntimeError(f"Could not download ShareGPT dataset: {e}") from e

    # Process ShareGPT conversations
    prompts: list[dict[str, Any]] = []
    for item in dataset:
        conversations = item.get("conversations", [])
        if not conversations:
            continue

        # Build chat from conversations
        chat = []
        for msg in conversations:
            role_map = {
                "human": "user",
                "gpt": "assistant",
                "system": "system",
                "user": "user",
                "assistant": "assistant",
            }
            role = role_map.get(msg.get("from", "").lower(), "user")
            content = msg.get("value", "")

            if content:
                chat.append({"role": role, "content": content})

        # Only use conversations with at least one user message
        if not any(msg["role"] == "user" for msg in chat):
            continue

        # For load testing, use only up to first user message
        first_user_idx = next(
            (i for i, msg in enumerate(chat) if msg["role"] == "user"),
            None
        )

        if first_user_idx is None:
            continue

        # Take messages up to and including first user message
        test_chat = chat[:first_user_idx + 1]
        user_prompt = test_chat[-1]["content"]

        try:
            num_tokens = len(
                tokenizer.apply_chat_template(
                    test_chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize ShareGPT prompt: {e}")
            continue

        if min_input_length <= num_tokens <= max_input_length:
            prompts.append({
                "prompt": user_prompt,
                "num_input_tokens": num_tokens,
                "conversation": test_chat,
            })

    logger.info(
        f"Loaded {len(prompts)} ShareGPT prompts "
        f"(filtered from {len(dataset)} with {min_input_length}-{max_input_length} tokens)"
    )

    if not prompts:
        raise ValueError(
            f"No ShareGPT prompts found within token range {min_input_length}-{max_input_length}"
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


def load_cnn_dailymail(
    tokenizer: PreTrainedTokenizerBase,
    min_input_length: int = 0,
    max_input_length: int = 2048,
    cache_dir: Path | None = None,
    num_samples: int = 1000,
    split: str = "test",
) -> list[dict[str, Any]]:
    """
    Load CNN/DailyMail dataset for summarization tasks.

    Args:
        tokenizer: Tokenizer for counting input tokens
        min_input_length: Minimum prompt length in tokens
        max_input_length: Maximum prompt length in tokens
        cache_dir: Directory to cache dataset
        num_samples: Number of samples to load (default: 1000)
        split: Dataset split ('train', 'validation', 'test')

    Returns:
        List of summarization prompt dictionaries
    """
    default_cache_dir = Path.cwd() / "datasets"
    cache_file = (cache_dir or default_cache_dir) / f"cnn_dailymail_{split}.jsonl"

    if cache_file.exists():
        logger.info(f"Loading cached CNN/DailyMail dataset from {cache_file}")
        with open(cache_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()][:num_samples]
    else:
        logger.info(f"Downloading CNN/DailyMail dataset ({split} split, {num_samples} samples)")
        url = f"https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/{split}.jsonl"

        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Load and cache
            dataset = []
            for i, line in enumerate(response.iter_lines()):
                if i >= num_samples:
                    break
                if line:
                    dataset.append(json.loads(line))

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")

            logger.info(f"Dataset cached to {cache_file}")

        except requests.RequestException as e:
            logger.error(f"Failed to download CNN/DailyMail: {e}")
            raise RuntimeError(f"Could not download CNN/DailyMail dataset: {e}") from e

    # Process for summarization prompts
    prompts: list[dict[str, Any]] = []
    for item in dataset:
        article = item.get("article", "")
        if not article:
            continue

        # Create summarization prompt
        prompt_text = f"Summarize the following article:\n\n{article}"

        chat = [{"role": "user", "content": prompt_text}]

        try:
            num_tokens = len(
            tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize CNN/DailyMail prompt: {e}")
            continue

        if min_input_length <= num_tokens <= max_input_length:
            prompts.append({
                "prompt": prompt_text,
                "num_input_tokens": num_tokens,
                "article": article,
                "highlights": item.get("highlights", ""),
            })

    logger.info(
        f"Loaded {len(prompts)} CNN/DailyMail prompts "
        f"(from {len(dataset)} with {min_input_length}-{max_input_length} tokens)"
    )

    if not prompts:
        raise ValueError(
            f"No CNN/DailyMail prompts found within token range {min_input_length}-{max_input_length}"
        )

    return prompts


def load_billsum(
    tokenizer: PreTrainedTokenizerBase,
    min_input_length: int = 1024,
    max_input_length: int = 8192,
    cache_dir: Path | None = None,
    num_samples: int = 500,
) -> list[dict[str, Any]]:
    """
    Load BillSum dataset for long context prefill testing.

    BillSum contains US Congressional and California state bills (long documents).
    Ideal for testing long context handling and prefill performance.

    Source: https://huggingface.co/datasets/FiscalNote/billsum

    Args:
        tokenizer: Tokenizer for counting input tokens
        min_input_length: Minimum prompt length (default: 1024 for long context)
        max_input_length: Maximum prompt length (default: 8192)
        cache_dir: Directory to cache dataset
        num_samples: Number of samples to load

    Returns:
        List of long-context prompt dictionaries
    """
    default_cache_dir = Path.cwd() / "datasets"
    cache_file = (cache_dir or default_cache_dir) / "billsum.jsonl"

    if cache_file.exists():
        logger.info(f"Loading cached BillSum dataset from {cache_file}")
        with open(cache_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()][:num_samples]
    else:
        logger.info(f"Downloading BillSum dataset from FiscalNote/billsum ({num_samples} samples)")
        # Using the test split for benchmarking (3,269 samples available)
        # Dataset is in Parquet format - we'll download it via the Parquet files API
        base_url = "https://huggingface.co/datasets/FiscalNote/billsum/resolve/main/data"
        url = f"{base_url}/test-00000-of-00001.parquet"

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            # Save parquet file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            # Read parquet file
            try:
                import pandas as pd
                df = pd.read_parquet(tmp_path)
                dataset = df.to_dict('records')[:num_samples]
            except ImportError:
                # Fallback: try using pyarrow directly
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(tmp_path)
                    df = table.to_pandas()
                    dataset = df.to_dict('records')[:num_samples]
                except ImportError:
                    raise RuntimeError(
                        "BillSum requires pandas or pyarrow to load. "
                        "Install with: pip install pandas pyarrow"
                    )
            finally:
                # Clean up temp file
                import os
                os.unlink(tmp_path)

            # Cache as JSONL for faster future loading
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")

            logger.info(f"Dataset cached to {cache_file}")

        except requests.RequestException as e:
            logger.error(f"Failed to download BillSum: {e}")
            raise RuntimeError(f"Could not download BillSum dataset: {e}") from e

    # Process for long-context prompts
    prompts: list[dict[str, Any]] = []
    for item in dataset:
        bill_text = item.get("text", "")
        if not bill_text:
            continue

        prompt_text = f"Summarize this legislative bill:\n\n{bill_text}"

        chat = [{"role": "user", "content": prompt_text}]

        try:
            num_tokens = len(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize BillSum prompt: {e}")
            continue

        if min_input_length <= num_tokens <= max_input_length:
            prompts.append({
                "prompt": prompt_text,
                "num_input_tokens": num_tokens,
                "summary": item.get("summary", ""),
                "title": item.get("title", ""),
            })

    logger.info(
        f"Loaded {len(prompts)} BillSum prompts "
        f"(from {len(dataset)} with {min_input_length}-{max_input_length} tokens)"
    )

    if not prompts:
        raise ValueError(
            f"No BillSum prompts found within token range {min_input_length}-{max_input_length}"
        )

    return prompts


def load_infinity_instruct(
    tokenizer: PreTrainedTokenizerBase,
    min_input_length: int = 512,
    max_input_length: int = 4096,
    cache_dir: Path | None = None,
    num_samples: int = 1000,
) -> list[dict[str, Any]]:
    """
    Load Infinity Instruct dataset for long context decode testing.

    Contains instructions that require long-form responses, ideal for testing
    generation capacity and decode performance.

    Args:
        tokenizer: Tokenizer for counting input tokens
        min_input_length: Minimum prompt length
        max_input_length: Maximum prompt length
        cache_dir: Directory to cache dataset
        num_samples: Number of samples to load

    Returns:
        List of long-decode prompt dictionaries
    """
    default_cache_dir = Path.cwd() / "datasets"
    cache_file = (cache_dir or default_cache_dir) / "infinity_instruct.jsonl"

    if cache_file.exists():
        logger.info(f"Loading cached Infinity Instruct dataset from {cache_file}")
        with open(cache_file) as f:
            dataset = [json.loads(line) for line in f if line.strip()][:num_samples]
    else:
        logger.info(f"Downloading Infinity Instruct dataset ({num_samples} samples)")
        # Using a subset focused on long-form instructions
        url = "https://huggingface.co/datasets/BAAI/Infinity-Instruct/resolve/main/data/train_7M.jsonl"

        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            dataset = []
            for i, line in enumerate(response.iter_lines()):
                if i >= num_samples * 2:  # Get extra to filter
                    break
                if line:
                    try:
                        dataset.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                for item in dataset:
                    f.write(json.dumps(item) + "\n")

            logger.info(f"Dataset cached to {cache_file}")

        except requests.RequestException as e:
            logger.error(f"Failed to download Infinity Instruct: {e}")
            raise RuntimeError(f"Could not download Infinity Instruct dataset: {e}") from e

    # Process instructions
    prompts: list[dict[str, Any]] = []
    for item in dataset:
        # Infinity Instruct format: {"instruction": "...", "output": "..."}
        instruction = item.get("instruction", "") or item.get("conversations", [{}])[0].get("value", "")
        if not instruction:
            continue

        chat = [{"role": "user", "content": instruction}]

        try:
            num_tokens = len(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize Infinity Instruct prompt: {e}")
            continue

        if min_input_length <= num_tokens <= max_input_length:
            prompts.append({
                "prompt": instruction,
                "num_input_tokens": num_tokens,
            })

        if len(prompts) >= num_samples:
            break

    logger.info(
        f"Loaded {len(prompts)} Infinity Instruct prompts "
        f"(from {len(dataset)} with {min_input_length}-{max_input_length} tokens)"
    )

    if not prompts:
        raise ValueError(
            f"No Infinity Instruct prompts found within token range {min_input_length}-{max_input_length}"
        )

    return prompts


def create_shared_prefix_dataset(
    tokenizer: PreTrainedTokenizerBase,
    shared_prefix: str,
    variable_suffixes: list[str],
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Create a dataset with shared prefix for prefix caching testing.

    All prompts share the same prefix (e.g., a long document or context)
    but have different suffixes (e.g., different questions about the document).

    This is ideal for testing prefix/KV cache efficiency.

    Args:
        tokenizer: Tokenizer for counting input tokens
        shared_prefix: Common prefix for all prompts
        variable_suffixes: List of different suffixes/questions
        system_prompt: Optional system prompt

    Returns:
        List of prompts with shared prefix

    Example:
        ```python
        prefix = "Given this 10,000 word document: [long text]"
        suffixes = [
            "What is the main topic?",
            "Who are the key people mentioned?",
            "Summarize the conclusion.",
        ]
        prompts = create_shared_prefix_dataset(tokenizer, prefix, suffixes)
        ```
    """
    prompts: list[dict[str, Any]] = []

    for suffix in variable_suffixes:
        full_prompt = f"{shared_prefix}\n\n{suffix}"

        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        chat.append({"role": "user", "content": full_prompt})

        try:
            num_tokens = len(
            tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize shared prefix prompt: {e}")
            continue

        prompts.append({
            "prompt": full_prompt,
            "num_input_tokens": num_tokens,
            "shared_prefix": shared_prefix,
            "suffix": suffix,
        })

    logger.info(f"Created {len(prompts)} shared-prefix prompts")
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
