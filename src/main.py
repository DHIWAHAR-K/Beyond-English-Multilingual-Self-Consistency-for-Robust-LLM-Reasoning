"""CLI entry point for cross-lingual consistency experiments."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.core import MLCRunner, MLXRunner, TorchMPSRunner
from src.profiler.latency import profile_latency
from src.tasks.parity import ParityTask
from src.tasks.reasoning import ReasoningTask
from src.tasks.translation import TranslationTask
from src.visualization.plots import plot_latency_cliff, plot_metric_bars


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    for model in config.get("models", []):
        if model["name"] == model_name:
            return model
    raise ValueError(f"Model '{model_name}' not found in config.")


def create_runner(backend: str):
    mapping = {
        "mlx": MLXRunner,
        "torch_mps": TorchMPSRunner,
        "mlc": MLCRunner,
    }
    if backend not in mapping:
        raise ValueError(f"Unsupported backend: {backend}")
    return mapping[backend]()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-lingual consistency evaluation CLI")
    parser.add_argument("--config", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--model", required=True, help="Model slug from config/models.yaml")
    parser.add_argument("--precision", default="int4", help="Quantization precision to evaluate")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profile", help="Run latency and memory profiling")

    subparsers.add_parser("parity", help="Run compression parity evaluation")

    reasoning = subparsers.add_parser("reasoning", help="Run CLC MGSM reasoning evaluation")
    reasoning.add_argument("--max-samples", type=int, default=5)
    reasoning.add_argument("--temperature", type=float, default=0.5)
    reasoning.add_argument("--baseline", action="store_true", help="Use T=0.0 baseline decoding")

    translate = subparsers.add_parser("translate", help="Run machine translation evaluation")
    translate.add_argument("--dataset", default="google/fleurs")
    translate.add_argument("--config-name", default="de_de")
    translate.add_argument("--split", default="train[:20]")
    translate.add_argument("--max-samples", type=int, default=20)
    translate.add_argument("--target-lang", default="German")
    translate.add_argument("--calibrate", choices=["autoround", "awq", "gptq"], default=None)

    plot = subparsers.add_parser("plot", help="Regenerate plots from saved JSON")
    plot.add_argument("--input", type=Path, required=True, help="Path to saved JSON results")
    plot.add_argument("--plot-type", choices=["latency", "metrics"], default="latency")

    return parser


def cmd_profile(args, model_cfg: Dict[str, Any], runner) -> None:
    runner.load_model(
        model_cfg["model_id"],
        args.precision,
        model_cfg.get("quantization_path"),
    )
    results = profile_latency(runner, model_cfg["model_id"])
    output_dir = args.output_dir / "latency" / args.model / args.precision
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable = {str(k): v for k, v in results.items()}
    json_path = output_dir / "profile.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=str)

    plot_path = args.output_dir / "plots" / f"latency_{args.model}_{args.precision}.png"
    plot_latency_cliff(results, plot_path)
    print(f"Saved profile to {json_path}")


def cmd_parity(args, runner) -> None:
    runner.load_model(
        find_model_config(load_config(args.config), args.model)["model_id"],
        args.precision,
        find_model_config(load_config(args.config), args.model).get("quantization_path"),
    )
    task = ParityTask(
        runner=runner,
        output_dir=args.output_dir / "parity",
        model_slug=args.model,
        precision=args.precision,
    )
    result = task.run()
    print(json.dumps(result, indent=2, default=str))


def cmd_reasoning(args, config: Dict[str, Any], runner) -> None:
    model_cfg = find_model_config(config, args.model)
    runner.load_model(model_cfg["model_id"], args.precision, model_cfg.get("quantization_path"))

    fallback_runner = None
    fallback_cfg = config.get("fallback_extractor")
    if fallback_cfg:
        fallback_runner = create_runner(fallback_cfg["backend"])
        fallback_runner.load_model(
            fallback_cfg["model_id"],
            fallback_cfg.get("precision", "int4"),
            fallback_cfg.get("quantization_path"),
        )

    task = ReasoningTask(
        runner=runner,
        output_dir=args.output_dir / "reasoning",
        model_slug=args.model,
        precision=args.precision,
        fallback_runner=fallback_runner,
    )
    result = task.run(
        max_samples=args.max_samples,
        temperature=args.temperature,
        baseline_temperature=args.baseline,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_translate(args, model_cfg: Dict[str, Any], runner) -> None:
    runner.load_model(model_cfg["model_id"], args.precision, model_cfg.get("quantization_path"))
    task = TranslationTask(
        runner=runner,
        output_dir=args.output_dir / "translation",
        model_slug=args.model,
        precision=args.precision,
    )
    result = task.run(
        dataset_name=args.dataset,
        config=args.config_name,
        split=args.split,
        max_samples=args.max_samples,
        target_lang=args.target_lang,
        calibrate=args.calibrate,
        model_id=model_cfg["model_id"],
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_plot(args) -> None:
    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    out_dir = args.output_dir / "plots"
    if args.plot_type == "latency":
        results = {int(k): v for k, v in payload.items()}
        plot_path = out_dir / f"{args.input.stem}_latency.png"
        plot_latency_cliff(results, plot_path)
    else:
        metrics = payload.get("metrics", payload)
        ci_bounds = payload.get("bootstrap", {})
        plot_path = out_dir / f"{args.input.stem}_metrics.png"
        plot_metric_bars(metrics, {k: ci_bounds for k in metrics}, plot_path)
    print(f"Saved plot to {plot_path}")


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model_cfg = find_model_config(config, args.model)
    runner = create_runner(model_cfg["backend"])

    if args.command == "profile":
        cmd_profile(args, model_cfg, runner)
    elif args.command == "parity":
        cmd_parity(args, runner)
    elif args.command == "reasoning":
        cmd_reasoning(args, config, runner)
    elif args.command == "translate":
        cmd_translate(args, model_cfg, runner)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()
