"""Public command-line entry point for HumanClawBench."""

from __future__ import annotations

import argparse
import json
import sys

from .assets import verify_bundled_assets, verify_weights
from .config import load_config


def _print_json(value) -> None:
    """Print a JSON value in deterministic human-readable form."""

    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser and all rollout/render/asset subcommands."""

    parser = argparse.ArgumentParser(prog="humanclaw-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="show a release profile")
    config.add_argument("profile", nargs="?", default="paper_fullval_v1")

    assets = commands.add_parser("assets", help="verify bundled assets and weights")
    assets.add_argument("--weights-root", default=None)
    assets.add_argument(
        "--weights-manifest",
        default="resources/weights/paper_fullval_v1.json",
    )

    prepare_hssd = commands.add_parser(
        "prepare-hssd",
        help="adapt an authorized HSSD val installation for HumanClawBench",
    )
    prepare_hssd.add_argument("--hssd-root", required=True)
    prepare_hssd.add_argument("--output", default="data/humanclaw-hssd-val41")
    prepare_hssd.add_argument(
        "--modifications",
        default="resources/hssd/humanclaw-hssd-val41",
    )
    prepare_hssd.add_argument(
        "--supplement",
        default=None,
        help=(
            "local supplement archive or extracted directory; when omitted, "
            "download the pinned Hugging Face asset"
        ),
    )
    prepare_hssd.add_argument("--skip-hash-check", action="store_true")

    run = commands.add_parser(
        "run",
        help="evaluate one episode, val100, fullval, or a custom episode list",
    )
    run.add_argument("--profile", default="paper_fullval_v1")
    run.add_argument("--model-config", required=True)
    run.add_argument(
        "--episodes",
        default="one",
        help="one, val100, fullval, or a JSON episode-list path",
    )
    run.add_argument("--scene-id", default=None)
    run.add_argument("--episode-id", default=None)
    run.add_argument("--object-category", default=None)
    run.add_argument("--scene-dataset-config", default=None)
    run.add_argument("--output", required=True)
    run.add_argument(
        "--gpus",
        default="auto",
        help=(
            "evaluation GPUs: auto or comma-separated IDs; exclude GPUs "
            "reserved by a local VLM server"
        ),
    )
    run.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="concurrent episode processes per evaluation GPU",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="skip episodes whose requested final artifacts already exist",
    )
    run.add_argument(
        "--video",
        action="store_true",
        help="also stream synchronized ego/exo MP4 files",
    )
    run.add_argument(
        "--metrics",
        action="store_true",
        help="also compute every benchmark metric and aggregate the run",
    )

    rollout = commands.add_parser("rollout", help="run one episode")
    rollout.add_argument("--profile", default="paper_fullval_v1")
    rollout.add_argument("--model-config", required=True)
    rollout.add_argument("--scene-id", required=True)
    rollout.add_argument("--episode-id", default=None)
    rollout.add_argument("--episode-index", type=int, default=0)
    rollout.add_argument("--object-category", default=None)
    rollout.add_argument("--scene-dataset-config", default=None)
    rollout.add_argument("--output-root", default=None)
    rollout.add_argument("--n-rollouts", type=int, default=1)
    rollout.add_argument("--device", default=None)
    rollout.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override the profile episode limit for a bounded runtime smoke",
    )
    rollout.add_argument(
        "--save-video",
        "--video",
        dest="save_video",
        action="store_true",
        help="stream synchronized ego/exo MP4 files during rollout",
    )
    rollout.add_argument(
        "--compute-metrics",
        "--metrics",
        dest="compute_metrics",
        action="store_true",
        help="compute the paper metrics and write one metrics.json",
    )

    batch = commands.add_parser("batch", help="run an episode slice in parallel")
    batch.add_argument("--profile", default="paper_fullval_v1")
    batch.add_argument("--model-config", required=True)
    batch.add_argument("--output-root", required=True)
    batch.add_argument("--max-parallel", type=int, default=1)
    batch.add_argument("--devices", default="")
    batch.add_argument("--offset", type=int, default=0)
    batch.add_argument("--limit", type=int, default=0)
    batch.add_argument(
        "--episode-list",
        default=None,
        help="JSON episode index, for example resources/benchmark/val100.json",
    )
    batch.add_argument(
        "--resume",
        action="store_true",
        help="skip episodes whose requested final artifacts are already complete",
    )
    batch.add_argument("--scene-dataset-config", default=None)
    batch.add_argument(
        "--save-video", "--video", dest="save_video", action="store_true"
    )
    batch.add_argument(
        "--compute-metrics",
        "--metrics",
        dest="compute_metrics",
        action="store_true",
    )

    metrics = commands.add_parser(
        "metrics",
        help="summarize completed metrics.json files without replaying episodes",
    )
    metrics.add_argument(
        "output_root",
        help="output tree recursively containing per-episode metrics.json files",
    )
    metrics.add_argument(
        "--json",
        dest="print_json",
        action="store_true",
        help="print the complete machine-readable summary instead of paper tables",
    )
    metrics.add_argument(
        "--write-json",
        action="store_true",
        help="also write <output-root>/metrics_summary.json",
    )

    render = commands.add_parser(
        "render",
        help="render ego/exo video from saved post-physics poses",
    )
    render.add_argument("--rollout-dir", required=True)
    render.add_argument("--output-dir", default=None)
    render.add_argument("--trajectory-path", default=None)
    render.add_argument("--preset", default="veryfast")
    render.add_argument("--crf", type=int, default=20)
    render.add_argument("--progress-every", type=int, default=100)
    render.add_argument("--force", action="store_true")

    render_batch = commands.add_parser(
        "render-batch",
        help="render saved trajectories in isolated parallel processes",
    )
    render_source = render_batch.add_mutually_exclusive_group(required=True)
    render_source.add_argument(
        "--input-root",
        help="recursively find standard trajectory_after.npz files",
    )
    render_source.add_argument(
        "--manifest",
        help="JSONL/JSON jobs with rollout_dir and optional trajectory_path",
    )
    render_batch.add_argument("--output-root", required=True)
    render_batch.add_argument("--max-parallel", type=int, default=1)
    render_batch.add_argument("--devices", default="")
    render_batch.add_argument("--preset", default="veryfast")
    render_batch.add_argument("--crf", type=int, default=20)
    render_batch.add_argument("--force", action="store_true")

    compose_video = commands.add_parser(
        "compose-video",
        help="combine saved ego/exo videos and per-step reasoning",
    )
    compose_video.add_argument("--rollout-dir", required=True)
    compose_video.add_argument("--output", default=None)
    compose_video.add_argument("--preset", default="veryfast")
    compose_video.add_argument("--crf", type=int, default=20)
    compose_video.add_argument("--threads", type=int, default=2)
    compose_video.add_argument("--force", action="store_true")

    compose_batch = commands.add_parser(
        "compose-video-batch",
        help="compose a completed rollout tree in parallel",
    )
    compose_batch.add_argument("--input-root", required=True)
    compose_batch.add_argument("--output-root", required=True)
    compose_batch.add_argument("--max-parallel", type=int, default=4)
    compose_batch.add_argument("--preset", default="veryfast")
    compose_batch.add_argument("--crf", type=int, default=20)
    compose_batch.add_argument("--threads", type=int, default=2)
    compose_batch.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point and return its process status."""

    args = _build_parser().parse_args(argv)
    if args.command == "config":
        _print_json(load_config(args.profile).data)
        return 0
    if args.command == "assets":
        rows = verify_bundled_assets()
        if args.weights_root:
            rows += verify_weights(args.weights_manifest, args.weights_root)
        _print_json(rows)
        return 0 if all(row["ok"] for row in rows) else 1
    if args.command == "prepare-hssd":
        from .hssd import prepare_hssd

        _print_json(
            prepare_hssd(
                args.hssd_root,
                args.output,
                modifications_root=args.modifications,
                supplement=args.supplement,
                verify_hashes=not args.skip_hash_check,
            )
        )
        return 0
    if args.command == "run":
        from .batch import resolve_devices, run_batch

        devices = resolve_devices(args.gpus)
        if not devices:
            raise RuntimeError(
                "No evaluation GPU was found. Set CUDA_VISIBLE_DEVICES or pass "
                "--gpus with a comma-separated GPU list."
            )
        if args.workers_per_gpu < 1:
            raise ValueError("--workers-per-gpu must be positive")

        episode_selection = str(args.episodes)
        selection_key = episode_selection.lower()
        if selection_key == "one":
            episode_list = None
            limit = 1
            require_unique = any(
                value is not None
                for value in (args.scene_id, args.episode_id, args.object_category)
            )
        elif selection_key == "val100":
            episode_list = "resources/benchmark/val100.json"
            limit = 0
            require_unique = False
        elif selection_key == "fullval":
            episode_list = None
            limit = 0
            require_unique = False
        else:
            episode_list = episode_selection
            limit = 0
            require_unique = False

        summary = run_batch(
            {
                "profile": load_config(args.profile),
                "model_config": args.model_config,
                "output_root": args.output,
                "devices": devices,
                "max_parallel": len(devices) * args.workers_per_gpu,
                "offset": 0,
                "limit": limit,
                "episode_list": episode_list,
                "scene_id": args.scene_id,
                "episode_id": args.episode_id,
                "object_category": args.object_category,
                "require_unique_episode": require_unique,
                "resume": args.resume,
                "scene_dataset_config": args.scene_dataset_config,
                "save_video": args.video,
                "compute_metrics": args.metrics,
            }
        )
        _print_json(summary)
        return 0 if summary["failed"] == 0 else 1
    if args.command == "rollout":
        from .evaluation.evaluator import HCFindNavInteractEvaluator

        config = vars(args).copy()
        config["profile"] = load_config(args.profile)
        config["model_config_path"] = args.model_config
        evaluator = HCFindNavInteractEvaluator(config)
        evaluator.check_config_valid()
        _print_json(evaluator.evaluate_main())
        return 0
    if args.command == "batch":
        from .batch import run_batch

        config = vars(args).copy()
        config["profile"] = load_config(args.profile)
        config["devices"] = tuple(
            item.strip() for item in args.devices.split(",") if item.strip()
        )
        summary = run_batch(config)
        _print_json(summary)
        return 0 if summary["failed"] == 0 else 1
    if args.command == "metrics":
        from .evaluation.metrics import (
            aggregate_metric_files,
            format_metric_summary,
        )

        summary = aggregate_metric_files(
            args.output_root,
            write_summary=args.write_json,
        )
        if int((summary.get("counts") or {}).get("episodes") or 0) == 0:
            print(
                f"No metrics.json files found under: {args.output_root}",
                file=sys.stderr,
            )
            return 1
        if args.print_json:
            _print_json(summary)
        else:
            print(format_metric_summary(summary))
        return 0
    if args.command == "render":
        from .rendering import render_saved_trajectory

        report = render_saved_trajectory(
            args.rollout_dir,
            args.output_dir,
            trajectory_path=args.trajectory_path,
            preset=args.preset,
            crf=args.crf,
            force=args.force,
            progress_every=args.progress_every,
        )
        _print_json(report)
        return 0
    if args.command == "render-batch":
        from .rendering.batch import (
            discover_render_jobs,
            load_render_jobs,
            render_saved_batch,
        )

        jobs = (
            discover_render_jobs(args.input_root, args.output_root)
            if args.input_root
            else load_render_jobs(args.manifest, args.output_root)
        )
        devices = tuple(
            item.strip() for item in args.devices.split(",") if item.strip()
        )
        summary = render_saved_batch(
            jobs,
            max_parallel=args.max_parallel,
            devices=devices,
            preset=args.preset,
            crf=args.crf,
            force=args.force,
        )
        _print_json(summary)
        return 0 if summary["failed"] == 0 else 1
    if args.command == "compose-video":
        from .rendering import compose_ego_exo_reasoning

        _print_json(
            compose_ego_exo_reasoning(
                args.rollout_dir,
                args.output,
                preset=args.preset,
                crf=args.crf,
                threads=args.threads,
                force=args.force,
            )
        )
        return 0
    if args.command == "compose-video-batch":
        from .rendering import compose_ego_exo_reasoning_batch

        summary = compose_ego_exo_reasoning_batch(
            args.input_root,
            args.output_root,
            max_parallel=args.max_parallel,
            preset=args.preset,
            crf=args.crf,
            threads=args.threads,
            force=args.force,
        )
        _print_json(summary)
        return 0 if summary["failed"] == 0 else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
