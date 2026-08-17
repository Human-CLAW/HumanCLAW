"""Train the concat-history base MotionDiT on 20-frame AMASS chunks."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader

from .datasets import ChunkTreeDataset, N_HISTORY, read_base_file_list
from .flow import sample_flow_matching_batch
from .profiles import load_training_profile
from .runtime import (
    load_torch_mapping,
    new_motion_dit,
    save_training_checkpoint,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small base-training interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--file-list",
        default=None,
        help="Optional newline-delimited relative pickle list.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _resolved_config(
    args: argparse.Namespace, profile: dict[str, Any]
) -> dict[str, Any]:
    """Merge explicit CLI overrides into the pinned base-training profile."""

    source = dict(profile["base_training"])
    overrides = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
    }
    source.update({key: value for key, value in overrides.items() if value is not None})
    source.update(
        {
            "chunk_root": str(Path(args.chunk_root).expanduser().resolve()),
            "file_list": args.file_list,
            "device": args.device,
            "checkpoint_every_epochs": args.checkpoint_every_epochs,
            "log_every_steps": args.log_every_steps,
            "seed": args.seed,
            "architecture": dict(profile["architecture"]),
        }
    )
    return source


def _restore_base_training(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Restore a checkpoint written by this refactored base trainer."""

    checkpoint = load_torch_mapping(path)
    if checkpoint.get("schema") != "humanclaw_base_training_v1":
        raise ValueError(f"Unsupported base-training resume checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    extra_state = checkpoint.get("extra_state", {})
    return int(extra_state.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def _write_inference_base(path: Path, model: torch.nn.Module) -> None:
    """Atomically export only the base tensors needed for later ControlNets."""

    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "schema": "humanclaw_motion_dit_state_v1",
            "state_dict": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    """Train, checkpoint, and export one base MotionDiT."""

    args = parse_args(argv)
    profile = load_training_profile(args.profile)
    config = _resolved_config(args, profile)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)

    torch.manual_seed(int(config["seed"]))
    relative_files = (
        None if args.file_list is None else read_base_file_list(args.file_list)
    )
    dataset = ChunkTreeDataset(
        args.chunk_root,
        relative_files=relative_files,
        normalize=bool(config["normalize"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=int(config["num_workers"]) > 0,
        drop_last=False,
    )
    device = torch.device(args.device)
    model = new_motion_dit(config["architecture"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"])
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = _restore_base_training(
            args.resume, model=model, optimizer=optimizer
        )

    max_epochs = int(config["max_epochs"])
    checkpoint_every = int(config["checkpoint_every_epochs"])
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every_epochs must be positive")
    started = time.perf_counter()
    model.train()
    for epoch in range(start_epoch, max_epochs):
        running_loss = 0.0
        batch_count = 0
        for batch in loader:
            history = batch["history"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            noisy_future, flow_time, target_velocity = sample_flow_matching_batch(
                future
            )
            concat_input = torch.cat([history, noisy_future], dim=1)
            condition = torch.zeros(
                history.shape[0],
                int(config["architecture"]["hidden_dim"]),
                device=device,
                dtype=history.dtype,
            )
            prediction = model(concat_input, flow_time, condition)[:, N_HISTORY:]
            loss = torch_functional.mse_loss(prediction, target_velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            optimizer.step()
            global_step += 1
            batch_count += 1
            running_loss += float(loss.item())
            if global_step <= 10 or global_step % int(config["log_every_steps"]) == 0:
                print(
                    f"epoch={epoch + 1}/{max_epochs} step={global_step} "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )
        mean_loss = running_loss / max(1, batch_count)
        should_checkpoint = (epoch + 1) % checkpoint_every == 0 or (
            epoch + 1 == max_epochs
        )
        if should_checkpoint:
            save_training_checkpoint(
                output / "checkpoints",
                step=global_step,
                schema="humanclaw_base_training_v1",
                model_state=model.state_dict(),
                optimizer=optimizer,
                config=config,
                extra_state={
                    "epoch": epoch + 1,
                    "mean_loss": mean_loss,
                    "sample_count": len(dataset),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        print(
            f"epoch={epoch + 1}/{max_epochs} mean_loss={mean_loss:.6f}", flush=True
        )

    _write_inference_base(output / "motion_dit.pt", model.cpu().eval())


if __name__ == "__main__":
    main()
