
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "openai-community/gpt2"
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    allow_patterns=[
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ],
)

print(f"Model downloaded to: {MODEL_DIR}")
