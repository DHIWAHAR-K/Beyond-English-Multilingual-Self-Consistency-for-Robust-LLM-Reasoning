"""
Validate all models defined in configs/models.yaml.

Steps
-----
1. Reasoning model guard   — ensure no reasoning-model IDs are configured
2. Metadata validation     — check required fields on every model
3. MLX load test           — attempt mlx_lm.load() for each eligible model
4. Tokenizer sanity check  — tokenize a probe string, assert non-empty int list
5. Memory report           — table of estimated RAM with large-model warnings
6. Final summary           — counts + total RAM estimate

Flags
-----
--model KEY     Test a single model by config key
--dry-run       Steps 1–2 only (no model loading)
--skip-large    Skip models with estimated_ram_gb > 40
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import sys
from pathlib import Path

import yaml

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
RESULTS_DIR = REPO_ROOT / "validation_check"

# ── rich ─────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console()
    _rich = True
except ImportError:
    console = None  # type: ignore[assignment]
    _rich = False

    def rprint(*args, **kwargs):  # type: ignore[misc]
        print(*args)


def _print(msg: str) -> None:
    if _rich:
        console.print(msg)
    else:
        print(msg)


# ── logging ──────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = RESULTS_DIR / f"model_validation_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _log(msg: str) -> None:
    """Print to console and write plain text to log file."""
    # Strip rich markup for log file
    import re
    plain = re.sub(r"\[.*?\]", "", msg)
    log.info(plain)
    _print(msg)


# ── mlx_lm import guard ──────────────────────────────────────────────────────
def _import_mlx_lm():
    try:
        import mlx_lm
        return mlx_lm
    except ImportError:
        _log(
            "[bold red]ERROR:[/bold red] mlx_lm is not installed.\n"
            "Install it with:  pip install mlx-lm\n"
            "or:               conda run -n scr pip install mlx-lm"
        )
        sys.exit(1)


# ── config loading ────────────────────────────────────────────────────────────
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        _log(f"[bold red]ERROR:[/bold red] Config not found: {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models", {})


# ── Step 1: Reasoning model guard ─────────────────────────────────────────────
def step1_reasoning_guard(models: dict) -> None:
    _log("\n[bold cyan]━━━ Step 1 — Reasoning Model Guard ━━━[/bold cyan]")
    blocked: list[str] = []
    for key, m in models.items():
        siblings = m.get("reasoning_siblings_to_avoid", [])
        mlx_id: str = m.get("mlx_id", "")
        for sibling in siblings:
            if sibling.lower() in mlx_id.lower():
                msg = (
                    f"BLOCKED: {key} is configured with a reasoning model ID "
                    f"'{mlx_id}'. This project requires non-reasoning models only."
                )
                _log(f"[bold red]❌ {msg}[/bold red]")
                log.error(msg)
                blocked.append(msg)

    if blocked:
        raise ValueError("\n".join(blocked))

    _log(
        "[bold green]✅ Reasoning model guard passed — "
        "no reasoning models detected in config[/bold green]"
    )


# ── Step 2: Metadata validation ───────────────────────────────────────────────
def step2_metadata(models: dict) -> list[str]:
    """Returns list of model keys that passed metadata validation."""
    _log("\n[bold cyan]━━━ Step 2 — Metadata Validation ━━━[/bold cyan]")
    passed: list[str] = []

    for key, m in models.items():
        errors: list[str] = []

        if m.get("is_reasoning_model") is not False:
            errors.append("is_reasoning_model must be explicitly false")
        if m.get("quantization") != "4bit":
            errors.append(f"quantization must be '4bit', got '{m.get('quantization')}'")
        ram = m.get("estimated_ram_gb")
        if not isinstance(ram, (int, float)) or ram <= 0:
            errors.append(f"estimated_ram_gb must be positive number, got '{ram}'")
        if not m.get("mlx_id", "").strip():
            errors.append("mlx_id must be a non-empty string")

        if errors:
            for e in errors:
                _log(f"  [red]❌ {key}: {e}[/red]")
        else:
            _log(f"  [green]✅ {key}: {m['name']}[/green]")
            passed.append(key)

    return passed


# ── Step 3 + 4: MLX load + tokenizer check ───────────────────────────────────
def step3_load_and_step4_tokenizer(
    models: dict,
    valid_keys: list[str],
    dry_run: bool,
    skip_large: bool,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (loaded_keys, skipped_keys, failed_keys).
    """
    _log("\n[bold cyan]━━━ Step 3 — MLX Load Test ━━━[/bold cyan]")

    if dry_run:
        _log("[yellow]  --dry-run active: skipping all model loads[/yellow]")
        return [], list(valid_keys), []

    mlx_lm = _import_mlx_lm()

    loaded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for key in valid_keys:
        m = models[key]
        name = m["name"]
        ram = m["estimated_ram_gb"]
        speed = m["estimated_speed_tok_per_sec"]
        mlx_id = m["mlx_id"]

        if m.get("requires_conversion", False):
            _log(f"  [yellow]⏭️  {name} — skipped (requires mlx_lm.convert first)[/yellow]")
            log.info(f"SKIPPED (conversion): {name}")
            skipped.append(key)
            continue

        if skip_large and ram > 40:
            _log(f"  [yellow]⏭️  {name} — skipped (--skip-large: {ram}GB > 40GB)[/yellow]")
            log.info(f"SKIPPED (large): {name} ({ram}GB)")
            skipped.append(key)
            continue

        try:
            _log(f"  [dim]Loading {name} ({mlx_id}) …[/dim]")
            model_obj, tokenizer = mlx_lm.load(mlx_id)
            _log(
                f"  [green]✅ {name} — loaded successfully "
                f"({ram}GB, ~{speed} tok/s)[/green]"
            )
            log.info(f"LOADED: {name} ({ram}GB, ~{speed} tok/s)")

            # ── Step 4: tokenizer sanity check ───────────────────────────
            _log(f"\n[bold cyan]━━━ Step 4 — Tokenizer Sanity: {name} ━━━[/bold cyan]")
            probe = "The quick brown fox"
            try:
                token_ids = tokenizer.encode(probe)
                assert isinstance(token_ids, list) and len(token_ids) > 0
                assert all(isinstance(t, int) for t in token_ids)
                _log(
                    f"  [green]✅ Tokenizer OK — "
                    f"'{probe}' → {len(token_ids)} tokens[/green]"
                )
                log.info(f"TOKENIZER OK: {name} ({len(token_ids)} tokens for probe)")
            except Exception as tok_err:
                _log(f"  [red]❌ Tokenizer FAILED: {tok_err}[/red]")
                log.error(f"TOKENIZER FAILED: {name}: {tok_err}")

            loaded.append(key)

        except Exception as err:
            _log(f"  [red]❌ {name} — FAILED: {err}[/red]")
            log.error(f"FAILED: {name}: {err}")
            failed.append(key)

        finally:
            # ── Unload model to free unified memory before next load ───────
            try:
                del model_obj
                del tokenizer
            except NameError:
                pass
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            _log(f"  [dim]  (memory released after {name})[/dim]")

    return loaded, skipped, failed


# ── Step 5: Memory report ─────────────────────────────────────────────────────
def step5_memory_report(
    models: dict,
    loaded: list[str],
    skipped: list[str],
    failed: list[str],
) -> None:
    _log("\n[bold cyan]━━━ Step 5 — Memory Report ━━━[/bold cyan]")

    status_map: dict[str, str] = {}
    for k in loaded:
        status_map[k] = "loaded"
    for k in skipped:
        status_map[k] = "skipped"
    for k in failed:
        status_map[k] = "failed"

    if _rich:
        table = Table(title="Model Memory Overview", show_lines=True)
        table.add_column("Model", style="bold", min_width=32)
        table.add_column("RAM (GB)", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Note", justify="center")

        for key, m in models.items():
            ram = m["estimated_ram_gb"]
            status = status_map.get(key, "not tested")
            note = "⚠️  LARGE — selective use only" if ram > 40 else ""

            if status == "loaded":
                status_str = "[green]✅ loaded[/green]"
            elif status == "skipped":
                status_str = "[yellow]⏭️  skipped[/yellow]"
            elif status == "failed":
                status_str = "[red]❌ failed[/red]"
            else:
                status_str = "[dim]— not tested[/dim]"

            table.add_row(m["name"], str(ram), status_str, note)

        console.print(table)
    else:
        header = f"{'Model':<40} {'RAM':>6}  {'Status':<12}  Note"
        _log(header)
        _log("-" * len(header))
        for key, m in models.items():
            ram = m["estimated_ram_gb"]
            status = status_map.get(key, "not tested")
            note = "⚠️  LARGE" if ram > 40 else ""
            _log(f"{m['name']:<40} {ram:>5}GB  {status:<12}  {note}")

    log.info("\nMEMORY REPORT:")
    for key, m in models.items():
        ram = m["estimated_ram_gb"]
        status = status_map.get(key, "not tested")
        flag = " [LARGE >40GB]" if ram > 40 else ""
        log.info(f"  {m['name']}: {ram}GB  {status}{flag}")


# ── Step 6: Final summary ─────────────────────────────────────────────────────
def step6_summary(
    models: dict,
    loaded: list[str],
    skipped: list[str],
    failed: list[str],
) -> None:
    _log("\n[bold cyan]━━━ Step 6 — Final Summary ━━━[/bold cyan]")

    total_ram = sum(m["estimated_ram_gb"] for m in models.values())
    ram_warning = " ⚠️  exceeds 128GB!" if total_ram > 128 else ""

    lines = [
        f"  Total models in config          : {len(models)}",
        f"  Reasoning model guard           : PASSED",
        f"  Models loaded successfully      : {len(loaded)}",
        f"  Models skipped (need conversion): {len(skipped)}",
        f"  Models failed                   : {len(failed)}",
        f"  Total estimated RAM (all models): {total_ram} GB{ram_warning}",
    ]

    for line in lines:
        _log(f"[bold]{line}[/bold]")
        log.info(line)

    _log(f"\n[dim]Full log written to: {log_path}[/dim]")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate models defined in configs/models.yaml"
    )
    parser.add_argument("--model", metavar="KEY", help="Test a single model by config key")
    parser.add_argument("--dry-run", action="store_true", help="Steps 1–2 only, no loading")
    parser.add_argument(
        "--skip-large", action="store_true", help="Skip models with estimated_ram_gb > 40"
    )
    args = parser.parse_args()

    _log(f"\n[bold white]{'='*60}[/bold white]")
    _log("[bold white]  Beyond-English — Model Validation Script[/bold white]")
    _log(f"[bold white]{'='*60}[/bold white]")
    _log(f"  Config : {CONFIG_PATH}")
    _log(f"  Log    : {log_path}")
    if args.dry_run:
        _log("  Mode   : [yellow]DRY RUN (Steps 1–2 only)[/yellow]")
    if args.skip_large:
        _log("  Mode   : [yellow]SKIP LARGE (>40GB models skipped)[/yellow]")

    all_models = load_config()

    # Filter to single model if --model flag given
    if args.model:
        if args.model not in all_models:
            _log(f"[bold red]ERROR:[/bold red] Model key '{args.model}' not found in config.")
            _log(f"  Available keys: {', '.join(all_models.keys())}")
            sys.exit(1)
        models = {args.model: all_models[args.model]}
        _log(f"  Scope  : single model → [bold]{args.model}[/bold]")
    else:
        models = all_models

    log.info(f"Run at: {timestamp}")
    log.info(f"Config: {CONFIG_PATH}")
    log.info(f"Models: {list(models.keys())}")
    log.info(f"dry_run={args.dry_run}, skip_large={args.skip_large}")

    # Step 1
    step1_reasoning_guard(models)

    # Step 2
    valid_keys = step2_metadata(models)

    # Steps 3 + 4
    loaded, skipped, failed = step3_load_and_step4_tokenizer(
        models, valid_keys, dry_run=args.dry_run, skip_large=args.skip_large
    )

    # Step 5
    step5_memory_report(models, loaded, skipped, failed)

    # Step 6
    step6_summary(models, loaded, skipped, failed)


if __name__ == "__main__":
    main()
