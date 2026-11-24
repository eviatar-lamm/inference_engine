"""Verify or regenerate English token IDs using OpenAI's unmodified encoder.py."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/openai/gpt-2/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", type=Path, help="local copy of OpenAI's encoder.py")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/gpt2")
    parser.add_argument("--write", action="store_true", help="regenerate the checked-in reference")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("official_gpt2_encoder", args.encoder)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    vocab_path = args.model_dir / "vocab.json"
    merges_path = args.model_dir / "merges.txt"
    merges = [tuple(line.split()) for line in merges_path.read_text(encoding="utf-8").splitlines()[1:]]
    encoder = official.Encoder(json.loads(vocab_path.read_text(encoding="utf-8")), merges)
    path = Path(__file__).with_name("tokenizer_reference.json")
    reference = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "source": SOURCE,
        "encoder_sha256": hashlib.sha256(args.encoder.read_bytes()).hexdigest(),
        "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "merges_sha256": hashlib.sha256(merges_path.read_bytes()).hexdigest(),
    }
    failures = []
    if not args.write and reference["reference"] != metadata:
        failures.append("Reference source or tokenizer file hashes differ")
    for case in reference["cases"]:
        if not case["text"].isascii():
            raise ValueError(f"Non-ASCII input in English-only case: {case['name']}")
        ids = encoder.encode(case["text"])
        if not args.write and case["ids"] != ids:
            failures.append(f"{case['name']}: expected {ids}, fixture has {case['ids']}")
        case["ids"] = ids
    if failures:
        raise SystemExit("\n".join(failures))
    if args.write:
        reference["reference"] = metadata
        path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"{'Wrote' if args.write else 'Verified'} {len(reference['cases'])} English cases against {SOURCE}")


if __name__ == "__main__":
    main()
