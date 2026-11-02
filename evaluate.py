"""Single-pass, reproducible text-translation evaluation (not STT or TTS)."""
import argparse
import hashlib
import json
import platform
import math
import time
from pathlib import Path

from src.evaluation_utils import load_pairs
from src.translator import Translator

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
    "--backend",
    choices=["m2m100", "marian"],
    default="marian",
)
    parser.add_argument("--direction", choices=["fr-en", "en-fr"], default="fr-en")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--english", type=Path, default=ROOT / "src.txt")
    parser.add_argument("--french", type=Path, default=ROOT / "ref.txt")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    english, french = load_pairs(args.english, args.french, args.n)
    source, target = args.direction.split("-")
    inputs, references = (french, english) if source == "fr" else (english, french)
    import sacrebleu
    import torch
    import transformers
    from sacrebleu.metrics import BLEU, CHRF
    from tqdm import tqdm
    translator = Translator(backend=args.backend, num_beams=args.beams,
                            model_path=args.model_path, device=args.device,
                            local_files_only=True if args.offline else None)
    loading_start = time.perf_counter()
    translator.warmup(source, target)
    translator.synchronize()
    load_seconds = time.perf_counter() - loading_start
    # Unique run directory prevents overwriting the baseline or earlier runs.
    run = args.output_dir / f"{args.backend}_{args.direction}_b{args.beams}_{time.time_ns()}"
    run.mkdir(parents=True, exist_ok=False)
    predictions, timings = [], []
    with (run / "predictions.txt").open("w", encoding="utf-8") as output:
        for sentence in tqdm(inputs, desc=f"{args.backend} {args.direction}", unit="sentence"):
            translator.synchronize()
            start = time.perf_counter()
            result = translator.translate(sentence, source, target)
            translator.synchronize()
            timings.append(time.perf_counter() - start)
            predictions.append(result)
            output.write(result.replace("\n", " ") + "\n")
            output.flush()
    bleu, chrf = BLEU(), CHRF()
    bleu_score = bleu.corpus_score(predictions, [references])
    chrf_score = chrf.corpus_score(predictions, [references])
    result = {
        "backend": args.backend, "direction": args.direction, "evaluated_pairs": args.n,
        "num_beams": args.beams, "device": str(translator.device),
        "model_path": str(translator.model_path) if args.backend == "m2m100" and translator.model_path else None,
        "bleu": bleu_score.score, "chrf2": chrf_score.score,
        "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
        "mean_translation_seconds": sum(timings) / len(timings),
        "p50_translation_seconds": sorted(timings)[len(timings)//2],
        "p95_translation_seconds": sorted(timings)[max(0, math.ceil(.95*len(timings))-1)],
        "model_load_seconds": load_seconds, "per_sentence_seconds": timings,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "sacrebleu": sacrebleu.__version__},
        "platform": platform.platform(),
        "input_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.english, args.french]},
        "scope": "First N aligned pairs; text translation only; timings exclude model loading but include first generation.",
    }
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nEvaluated pairs: {args.n}\nsacreBLEU version: {sacrebleu.__version__}")
    print("\nBLEU:", bleu_score, "\nSignature:", bleu.get_signature())
    print("\nchrF:", chrf_score, "\nSignature:", chrf.get_signature())
    print(f"\nMean translation: {result['mean_translation_seconds']:.3f}s\nResults: {run}")


if __name__ == "__main__":
    main()
