# Datasets Directory

This directory stores downloaded and cached datasets for LLM load testing.

## Cached Datasets

Datasets are automatically downloaded on first use and cached here:

| File | Dataset | Size | Source |
|------|---------|------|--------|
| `databricks-dolly-15k.jsonl` | Databricks Dolly | ~13 MB | HuggingFace |
| `sharegpt.jsonl` | ShareGPT Conversations | ~150 MB | HuggingFace |
| `cnn_dailymail_test.jsonl` | CNN/DailyMail | ~50 MB | HuggingFace |
| `billsum.jsonl` | BillSum | ~30 MB | HuggingFace |
| `infinity_instruct.jsonl` | Infinity Instruct | ~100 MB | HuggingFace |

## Management

### Re-download a Dataset
```bash
# Delete the cached file and it will re-download on next use
rm datasets/sharegpt.jsonl
```

### Clear All Datasets
```bash
# Remove all cached datasets
rm datasets/*.jsonl
```

### Disk Space
Approximate total disk usage: **~350-400 MB** for all datasets combined.

## Custom Datasets

You can also place your custom datasets here:

```bash
# Your custom prompts
datasets/
├── my_custom_prompts.jsonl
├── domain_specific.jsonl
└── production_samples.json
```

Then load them:
```python
from llm_locust.utils import load_custom_prompts
from pathlib import Path

prompts = load_custom_prompts(
    tokenizer=tokenizer,
    prompts_file=Path("datasets/my_custom_prompts.jsonl"),
)
```

## Notes

- All dataset files are **gitignored** (won't be committed)
- First download may take time depending on dataset size
- Subsequent loads are instant (uses cached files)
- Files are stored as JSONL for consistency and streaming compatibility

