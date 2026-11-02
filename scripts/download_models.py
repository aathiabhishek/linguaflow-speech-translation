"""Download translation weights ahead of offline use."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["m2m100", "marian"], default="m2m100")
    parser.add_argument("--output", type=Path, default=ROOT / "models/m2m100_418M")
    args = parser.parse_args()
    from transformers import (M2M100ForConditionalGeneration, M2M100Tokenizer,
                              MarianMTModel, MarianTokenizer)
    if args.backend == "m2m100":
        name = "facebook/m2m100_418M"
        tokenizer = M2M100Tokenizer.from_pretrained(name)
        model = M2M100ForConditionalGeneration.from_pretrained(name)
        args.output.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(args.output)
        model.save_pretrained(args.output)
        print(f"Saved model and tokenizer: {args.output}")
    else:
        for pair in ["en-fr", "fr-en"]:
            name = f"Helsinki-NLP/opus-mt-{pair}"
            MarianTokenizer.from_pretrained(name)
            MarianMTModel.from_pretrained(name)
        print("Both Marian models are in the Hugging Face cache.")


if __name__ == "__main__":
    main()
