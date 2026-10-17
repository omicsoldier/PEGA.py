"""
pega.cli
========
Command-line interface for PEGA.py.
"""

from __future__ import annotations

import os

# Must be set BEFORE tensorflow is imported anywhere in this process.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import sys
from pathlib import Path

from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.table import Table

# Force line-buffering even when stdout/stderr are redirected to a file
# (e.g. `nohup PEGA screen ... &> log.txt`). Without this, Python switches
# to full block buffering for non-TTY output, so progress lines can sit
# unflushed for a long time even while PEGA is actively working.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

console = Console()

# ---------------------------------------------------------------------------
# Banner — lazy to avoid AttributeError on import
# ---------------------------------------------------------------------------

_LOGO = r"""
 ██████╗ ███████╗ ██████╗  █████╗
 ██╔══██╗██╔════╝██╔════╝ ██╔══██╗
 ██████╔╝█████╗  ██║  ███╗███████║
 ██╔═══╝ ██╔══╝  ██║   ██║██╔══██║
 ██║     ███████╗╚██████╔╝██║  ██║   ██╗ █████╗ ██╗  ██╗
 ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝ ██╔══██╗╚██╗██╔╝
                                         █████╔╝  ╚███╔╝
                                         ██╔══╝   ██╔╝
                                         ╚═╝      ╚═╝    """

_CHAIN_COLORS = ["cyan", "magenta", "blue", "green"]


def _peptide_chain(width: int) -> str:
    """A small colored "peptide backbone" decoration: ●─●─●─●─...

    Purely cosmetic — a nod to what PEGA actually works on.
    """
    n = max(1, width // 2)
    residues = [
        f"[{_CHAIN_COLORS[i % len(_CHAIN_COLORS)]}]●[/{_CHAIN_COLORS[i % len(_CHAIN_COLORS)]}]"
        for i in range(n)
    ]
    return "[dim]─[/dim]".join(residues)


def _get_version() -> str:
    try:
        import pega as _pega
        return _pega.__version__
    except AttributeError:
        return "0.1.0"


def _print_banner() -> None:
    """Print the PEGA banner, adapting to the terminal width.

    Full logo on normal-or-wider terminals; a compact one-liner on narrow
    ones (e.g. a tmux pane on a cluster node), where the block-letter
    logo would otherwise wrap and look broken.
    """
    version = _get_version()
    if console.width >= 60:
        console.print(_LOGO, style="bold cyan", highlight=False)
        console.print(f"  {_peptide_chain(min(console.width - 4, 56))}")
        console.print(
            f"  [bold]Peptide Evolution via Genetic Algorithm[/bold]  "
            f"[green]v{version}[/green]"
        )
        console.print("  [dim]https://github.com/omicsoldier/PEGA.py[/dim]\n")
    else:
        console.print(
            f"[bold cyan]PEGA[/bold cyan] [green]v{version}[/green] "
            f"[dim]— Peptide Evolution via Genetic Algorithm[/dim]\n"
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suppress_warnings() -> None:
    """Suppress TensorFlow, absl, and HuggingFace informational output."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"]        = "false"
    os.environ["HF_HUB_VERBOSITY"]              = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]  = "1"
    os.environ["TRANSFORMERS_VERBOSITY"]         = "error"
    import warnings
    import logging
    warnings.filterwarnings("ignore")
    for logger_name in (
        "tensorflow", "absl", "transformers",
        "huggingface_hub", "huggingface_hub.utils._http",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _fasta_stats(fasta_path: Path) -> dict:
    """Return basic statistics for a FASTA file."""
    from Bio import SeqIO
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    lengths = [len(r.seq) for r in records]
    return {
        "n": len(records),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    if not args.quiet:
        _print_banner()
    if args.ensembles:
        from pega.ensemble import list_ensembles, ENSEMBLES
        table = Table(box=SIMPLE_HEAVY, show_edge=False)
        table.add_column("Name", style="bold cyan")
        table.add_column("Method")
        table.add_column("Pred", justify="right")
        table.add_column("Performance", justify="right")
        table.add_column("Deploy")
        table.add_column("Description")
        for e in list_ensembles():
            m = e["metrics"]
            if e["objective"] == "screening":
                perf = f"BEDROC {m['bedroc']:.3f}"
            else:
                perf = f"MCC {m['mcc_nested_cv']:.3f}"
            table.add_row(
                e["name"], e["method"], str(e["n_predictors"]), perf,
                "yes" if e["deploy"] else "[bold red]no[/bold red]",
                e["description"][:40],
            )
        console.print(table)
        console.print(f"\n  [bold]{len(ENSEMBLES)}[/bold] ensembles available.")
        for e in list_ensembles():
            if e["deploy_note"]:
                console.print(f"  [dim]{e['name']}: {e['deploy_note']}[/dim]")
    else:
        from pega.registry import registry
        table = Table(box=SIMPLE_HEAVY, show_edge=False)
        table.add_column("ID", justify="right")
        table.add_column("Predictor", style="bold")
        table.add_column("Status")
        table.add_column("Type")
        table.add_column("Description")
        for cls in registry.list_all():
            status = ("[green]available[/green]" if cls.is_available()
                       else "[red]unavailable[/red]")
            table.add_row(
                str(cls.predictor_id), cls.display_name, status,
                cls.category, cls.description,
            )
        console.print(table)
        n_ok = len(registry.list_available())
        n_all = len(registry.list_all())
        console.print(
            f"\n  [bold]{n_ok}[/bold] of [bold]{n_all}[/bold] predictors available."
        )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    if args.quiet:
        _suppress_warnings()

    from pega.utils import calculate_scores

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}", file=sys.stderr)
        return 1

    predictors = args.predictors or None

    if not args.quiet:
        _print_banner()

    # Parse --ensembles argument
    ensemble_arg = args.ensembles
    if ensemble_arg is None:
        ensemble_names = None          # all
    elif len(ensemble_arg) == 1 and ensemble_arg[0].lower() == "none":
        ensemble_names = []            # skip all
    else:
        ensemble_names = [
            f"ensemble_{e}_score" if not e.startswith("ensemble_") else e
            for e in ensemble_arg
        ]

    try:
        calculate_scores(
            fasta_path=fasta,
            predictor_names=predictors,
            exclude_predictors=args.exclude or None,
            export_tsv=args.out,
            jobs=args.jobs,
            quiet=args.quiet,
            ensemble_names=ensemble_names,
            no_individual_scores=args.no_individual_scores,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    """Chunked screening for large FASTA files."""
    if args.quiet:
        _suppress_warnings()

    from pega.utils import screen_sequences

    fasta = Path(args.fasta)
    if not fasta.exists():
        print(f"Error: FASTA file not found: {fasta}", file=sys.stderr)
        return 1

    if not args.quiet:
        _print_banner()

    ensemble_names = None
    if hasattr(args, "ensembles") and args.ensembles:
        ensemble_names = [
            f"ensemble_{e}_score" if not e.startswith("ensemble_") else e
            for e in args.ensembles
        ]

    try:
        screen_sequences(
            fasta_path=fasta,
            output_dir=getattr(args, "dir", None),
            chunk_size=args.chunk,
            exclude_predictors=args.exclude or None,
            ensemble_names=ensemble_names,
            no_individual_scores=getattr(args, "no_individual_scores", False),
            jobs=args.jobs,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from pega.preprocess.validator import validate_fasta
    if not args.quiet:
        _print_banner()
    ok, _ = validate_fasta(Path(args.fasta), verbose=True)
    return 0 if ok else 1


def cmd_setup(args: argparse.Namespace) -> int:
    from pega.setup_envs import print_status, setup_all
    if not args.quiet:
        _print_banner()
    if args.status:
        print("Environment status:\n")
        print_status()
        return 0
    setup_all(envs=args.envs or None, include_r=not args.no_r, force=args.force)
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    from pega.download_models import download_models
    if not args.quiet:
        _print_banner()
    download_models(model_dir=args.dir, overwrite=args.overwrite)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PEGA",
        description="PEGA.py — Peptide Evolution via Genetic Algorithm\n"
                    "Multi-predictor scoring for antimicrobial peptide research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  PEGA list\n"
            "  PEGA validate --fasta sequences.fasta\n"
            "  PEGA score --fasta sequences.fasta\n"
            "  PEGA score --fasta sequences.fasta --predictors AMPnet modlAMP_RF\n"
            "  PEGA score --fasta sequences.fasta --out results.tsv --jobs 4\n"
            "  PEGA score --fasta sequences.fasta --quiet\n"
            "  PEGA score --fasta sequences.fasta --ensembles AMP_class\n"
            "  PEGA score --fasta sequences.fasta --ensembles AMP_screen ABP_screen --no-individual-scores\n"
            "  PEGA setup --status\n"
            "  PEGA download-models\n"
        ),
    )
    parser.add_argument("--version", action="version",
                        version="PEGA.py 0.1.0")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # list
    lp = sub.add_parser("list", help="Show all predictors and their availability.")
    lp.add_argument("--ensembles", action="store_true", default=False,
                    help="Show available ensemble scores instead of predictors.")
    lp.add_argument("-q", "--quiet", action="store_true", default=False)

    # score
    sp = sub.add_parser("score", help="Score sequences in a FASTA file.")
    sp.add_argument("--fasta", "-f", required=True, metavar="FILE")
    sp.add_argument("--exclude", "-x", nargs="+", metavar="NAME",
                    help="Predictor names to leave out. Applied after "
                         "--predictors, so the two combine. Default: exclude none.")
    sp.add_argument("--predictors", "-p", nargs="+", metavar="NAME",
                    help="Predictor names (run 'PEGA list' for names). Default: all.")
    sp.add_argument("--out", "-o", metavar="FILE",
                    help="Output TSV file path. Default: results_PEGA/PEGA_results_<timestamp>.tsv")
    sp.add_argument("--jobs", "-j", type=int, default=1, metavar="N",
                    help="Parallel workers. Use -1 for 75%% of available CPU threads. Default: 1 (sequential).")
    sp.add_argument("--ensembles", "-e", nargs="+", metavar="NAME",
                    help="Only compute specific ensembles, e.g. AMP_class ABP_screen "
                         "(activity AMP/AVP/AFP/APP/ABP + _class or _screen). "
                         "Pass 'none' to skip all ensembles. Default: all.")
    sp.add_argument("--no-individual-scores", action="store_true", default=False,
                    help="Output only ensemble columns.")
    sp.add_argument("-q", "--quiet", action="store_true", default=False,
                    help="Suppress banner and framework warnings.")

    # screen
    scrp = sub.add_parser(
        "screen",
        help="Score a large FASTA file in chunks (100k–1M+ sequences).",
        description=(
            "Chunked screening mode for large datasets. "
            "Results are written to TSV progressively — "
            "safe to interrupt and resume."
        ),
    )
    scrp.add_argument("--fasta", "-f", required=True, metavar="FILE")
    scrp.add_argument("--dir", metavar="PATH",
                      help="Output directory. Default: PEGA_screen_YYYYMMDD_HHMMSS/ "
                           "Contains: names.tsv, rejected.tsv, rejected.fasta, scores.tsv")
    scrp.add_argument("--chunk", type=int, default=50_000, metavar="N",
                      help="Sequences per chunk (default: 50,000). Reduce to 10,000 if RAM < 16 GB. "
                           "Increase to 50,000 if you have ≥32 GB RAM.")
    scrp.add_argument("--exclude", "-x", nargs="+", metavar="NAME",
                      help="Predictor names to leave out. Default: exclude none.")
    scrp.add_argument("--ensembles", "-e", nargs="+", metavar="NAME",
                      help="Only compute specific ensembles, e.g. AMP_screen ABP_screen "
                           "(activity AMP/AVP/AFP/APP/ABP + _class or _screen). "
                           "Auto-selects required predictors.")
    scrp.add_argument("--no-individual-scores", action="store_true", default=False,
                      help="Output only ensemble columns.")
    scrp.add_argument("--jobs", "-j", type=int, default=-1, metavar="N",
                      help="Parallel workers. Default: -1 (75%% of CPU threads).")
    scrp.add_argument("-q", "--quiet", action="store_true", default=False)

    # validate
    vp = sub.add_parser("validate", help="Validate a FASTA file before scoring.")
    vp.add_argument("--fasta", "-f", required=True, metavar="FILE")
    vp.add_argument("-q", "--quiet", action="store_true", default=False)

    # setup
    ep = sub.add_parser("setup", help="Create conda environments.")
    ep.add_argument("--envs", nargs="+", metavar="NAME")
    ep.add_argument("--no-r", action="store_true", default=False)
    ep.add_argument("--force", action="store_true", default=False)
    ep.add_argument("--status", action="store_true", default=False)
    ep.add_argument("-q", "--quiet", action="store_true", default=False)

    # download-models
    dp = sub.add_parser("download-models", help="Download pre-trained model weights.")
    dp.add_argument("--dir", metavar="PATH", default=None)
    dp.add_argument("--overwrite", action="store_true", default=False)
    dp.add_argument("-q", "--quiet", action="store_true", default=False)

    return parser


_HANDLERS = {
    "list":            cmd_list,
    "score":           cmd_score,
    "screen":          cmd_screen,
    "validate":        cmd_validate,
    "setup":           cmd_setup,
    "download-models": cmd_download_models,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(_HANDLERS[args.command](args))


if __name__ == "__main__":
    main()
