"""Train any released skill ControlNet with one shared, paper-aligned loop."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional

from humanclaw_bench.motion.networks.noncond_ctrl_dit import NonCondCtrlDiT
from humanclaw_bench.motion.networks.side_walk_ctrl_dit_fourier import (
    SideWalkCtrlDiTFourier,
)
from humanclaw_bench.motion.networks.sit_ctrl_dit import SitCtrlDiT
from humanclaw_bench.motion.networks.step_climb_down_ctrl_dit import (
    StepClimbDownCtrlDiT,
)
from humanclaw_bench.motion.networks.step_climb_up_ctrl_dit import StepClimbUpCtrlDiT
from humanclaw_bench.motion.networks.turn_ctrl_dit import TurnCtrlDiT
from humanclaw_bench.motion.networks.walk_forward_ctrl_dit_fourier import (
    WalkForwardCtrlDiTFourier,
)
from humanclaw_bench.motion.networks.walk_forward_ctrl_dit_fourier_xzyaw import (
    WalkForwardCtrlDiTFourierXZYaw,
)
from humanclaw_bench.paths import resolve_release_path

from .datasets import N_HISTORY, load_skill_arrays, repeat_skill_arrays
from .flow import sample_flow_matching_batch
from .profiles import load_training_profile, selected_skill_profile
from .runtime import (
    configure_cuda_training,
    count_parameters,
    format_duration,
    load_base_motion_dit,
    load_resume_checkpoint,
    make_batch_indices,
    patch_fast_kernels,
    resolve_dtype,
    save_training_checkpoint,
    trainable_parameters,
    write_json,
)

CONTROL_NETWORKS = {
    "WalkForwardCtrlDiTFourierXZYaw": WalkForwardCtrlDiTFourierXZYaw,
    "SideWalkCtrlDiTFourier": SideWalkCtrlDiTFourier,
    "WalkForwardCtrlDiTFourier": WalkForwardCtrlDiTFourier,
    "TurnCtrlDiT": TurnCtrlDiT,
    "StepClimbUpCtrlDiT": StepClimbUpCtrlDiT,
    "StepClimbDownCtrlDiT": StepClimbDownCtrlDiT,
    "NonCondCtrlDiT": NonCondCtrlDiT,
    "SitCtrlDiT": SitCtrlDiT,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a compact interface whose paper defaults live in one JSON profile."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--chunk-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    return parser.parse_args(argv)


def _build_control_model(
    base: torch.nn.Module, architecture: dict[str, Any], config: dict[str, Any]
) -> torch.nn.Module:
    """Instantiate the profile-selected ControlNet around a frozen base."""

    network_name = str(config["network"])
    try:
        network_class = CONTROL_NETWORKS[network_name]
    except KeyError as error:
        raise ValueError(f"Unsupported control network: {network_name}") from error
    arguments: dict[str, Any] = {
        "base_dit": base,
        "hidden_dim": int(architecture["hidden_dim"]),
    }
    if "n_freqs" in config:
        arguments["n_freqs"] = int(config["n_freqs"])
    return network_class(**arguments)


def _network_condition(
    skill: str, raw_condition: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    """Apply the exact condition normalization used during paper training."""

    if skill == "stop":
        return ()
    if raw_condition is None:
        raise ValueError(f"Skill {skill} requires a numeric condition")
    if skill == "turn":
        return (raw_condition / 75.0,)
    if skill in {"step_climb_up", "step_climb_down"}:
        normalized = torch.cat(
            [
                4.0 * raw_condition[:, :1] - 1.0,
                (raw_condition[:, 1:2] - 0.5) / 0.3,
            ],
            dim=-1,
        )
        return (normalized,)
    return (raw_condition,)


def _autocast_context(device: torch.device, config: dict[str, Any]):
    """Return the BF16 autocast context used by every final FP32 skill run."""

    if config["parameter_dtype"] != "fp32":
        return contextlib.nullcontext()
    if config.get("autocast_dtype") != "bf16":
        return contextlib.nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _write_control_state(path: Path, model: torch.nn.Module) -> None:
    """Export only trainable control tensors, omitting the repeated frozen base."""

    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("base_dit.")
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {"schema": "humanclaw_control_state_v1", "state_dict": state}, temporary
    )
    os.replace(temporary, path)


def _merged_config(
    args: argparse.Namespace,
    *,
    profile: dict[str, Any],
    skill_name: str,
    skill_config: dict[str, Any],
    manifest: Path,
) -> dict[str, Any]:
    """Record the exact effective configuration before training starts."""

    config = dict(skill_config)
    override_pairs = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "log_every_steps": args.log_every_steps,
    }
    config.update(
        {key: value for key, value in override_pairs.items() if value is not None}
    )
    config.update(
        {
            "skill": skill_name,
            "architecture": dict(profile["architecture"]),
            "manifest": str(manifest),
            "chunk_root": str(Path(args.chunk_root).expanduser().resolve()),
            "base_checkpoint": str(
                Path(args.base_checkpoint).expanduser().resolve()
            ),
            "device": args.device,
            "max_samples": args.max_samples,
            "compile_mode": "none" if args.no_compile else config["compile_mode"],
        }
    )
    return config


def main(argv: list[str] | None = None) -> None:
    """Train one ControlNet and emit both resumable and inference-only states."""

    args = parse_args(argv)
    profile = load_training_profile(args.profile)
    skill_name, skill_config = selected_skill_profile(profile, args.skill)
    manifest = (
        resolve_release_path(args.manifest)
        if args.manifest is not None
        else resolve_release_path(skill_config["manifest"])
    )
    config = _merged_config(
        args,
        profile=profile,
        skill_name=skill_name,
        skill_config=skill_config,
        manifest=manifest,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)

    configure_cuda_training()
    torch.manual_seed(int(config["seed"]))
    arrays = load_skill_arrays(
        skill_name,
        chunk_root=args.chunk_root,
        manifest_path=manifest,
        max_samples=args.max_samples,
    )
    if args.max_samples is None and len(arrays.chunks) != int(config["expected_samples"]):
        raise ValueError(
            f"{skill_name} manifest contains {len(arrays.chunks)} samples; "
            f"paper profile requires {config['expected_samples']}"
        )
    original_sample_count = int(arrays.chunks.shape[0])
    arrays = repeat_skill_arrays(arrays, int(config["repeat_factor"]))

    device = torch.device(args.device)
    data_dtype = resolve_dtype(str(config["data_dtype"]))
    parameter_dtype = resolve_dtype(str(config["parameter_dtype"]))
    chunks = torch.from_numpy(arrays.chunks).to(
        device=device, dtype=data_dtype, non_blocking=True
    )
    history_all = chunks[:, :N_HISTORY].contiguous()
    future_all = chunks[:, N_HISTORY:].contiguous()
    condition_all = (
        None
        if arrays.condition is None
        else torch.from_numpy(arrays.condition)
        .to(device=device, dtype=data_dtype, non_blocking=True)
        .contiguous()
    )
    del chunks, arrays

    base = load_base_motion_dit(args.base_checkpoint, profile["architecture"])
    raw_model = _build_control_model(base, profile["architecture"], config).to(
        device=device, dtype=parameter_dtype
    )
    patched_attention = patch_fast_kernels(
        raw_model, use_sdpa=bool(config["use_sdpa"])
    )
    parameters = trainable_parameters(raw_model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        fused=device.type == "cuda",
    )
    start_step = 0
    if args.resume:
        start_step = load_resume_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(config["learning_rate"])

    train_model: torch.nn.Module = raw_model
    if config["compile_mode"] != "none":
        compile_mode = (
            None if config["compile_mode"] == "default" else config["compile_mode"]
        )
        train_model = torch.compile(
            raw_model, mode=compile_mode, fullgraph=False, dynamic=False
        )
    train_model.train()
    total_steps = int(config["max_steps"])
    if start_step >= total_steps:
        raise ValueError(
            f"Resume step {start_step} is not below requested max_steps {total_steps}"
        )
    sample_count = int(history_all.shape[0])
    batch_size = int(config["batch_size"])
    checkpoint_every = int(config["checkpoint_every_steps"])
    log_every = int(config["log_every_steps"])
    print(
        f"skill={skill_name} original_samples={original_sample_count} "
        f"training_samples={sample_count} trainable={count_parameters(parameters):,} "
        f"attention_modules={patched_attention}",
        flush=True,
    )

    log_path = output / "train_log.jsonl"
    started = time.perf_counter()
    steady_started: float | None = None
    steady_step = start_step
    with log_path.open("a", encoding="utf-8") as log_handle:
        for zero_based_step in range(start_step, total_steps):
            indices = make_batch_indices(sample_count, batch_size, device)
            history = history_all.index_select(0, indices)
            future = future_all.index_select(0, indices)
            raw_condition = (
                None if condition_all is None else condition_all.index_select(0, indices)
            )
            optimizer.zero_grad(set_to_none=True)
            noisy_future, flow_time, target_velocity = sample_flow_matching_batch(
                future
            )
            with _autocast_context(device, config):
                concat_input = torch.cat([history, noisy_future], dim=1).to(
                    dtype=parameter_dtype
                )
                condition = (
                    None
                    if raw_condition is None
                    else raw_condition.to(dtype=parameter_dtype)
                )
                prediction = train_model(
                    concat_input,
                    flow_time.to(dtype=parameter_dtype),
                    *_network_condition(skill_name, condition),
                )[:, N_HISTORY:]
            # The historical fast trainers left autocast before evaluating
            # the FP32 loss.  Keeping that boundary avoids changing gradient
            # numerics while consolidating their eight duplicated loops.
            loss = torch_functional.mse_loss(
                prediction.float(), target_velocity.float()
            )
            loss.backward()
            optimizer.step()
            step = zero_based_step + 1
            if step == int(config["warmup_steps"]):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                steady_started = time.perf_counter()
                steady_step = step

            should_log = step <= 10 or step % log_every == 0 or step == total_steps
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if steady_started is None or step <= steady_step:
                    steps_per_second = (step - start_step) / max(elapsed, 1e-6)
                else:
                    steps_per_second = (step - steady_step) / max(
                        time.perf_counter() - steady_started, 1e-6
                    )
                record = {
                    "step": step,
                    "loss": float(loss.item()),
                    "steps_per_second": steps_per_second,
                    "samples_per_second": steps_per_second * batch_size,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (total_steps - step)
                    / max(steps_per_second, 1e-6),
                }
                print(
                    f"step={step}/{total_steps} loss={record['loss']:.6f} "
                    f"steps/s={steps_per_second:.2f} "
                    f"eta={format_duration(record['eta_seconds'])}",
                    flush=True,
                )
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                log_handle.flush()

            should_checkpoint = (
                step == 1 or step % checkpoint_every == 0 or step == total_steps
            )
            if should_checkpoint:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                save_training_checkpoint(
                    output / "checkpoints",
                    step=step,
                    schema="humanclaw_control_training_v1",
                    model_state=raw_model.state_dict(),
                    optimizer=optimizer,
                    config=config,
                    extra_state={
                        "original_sample_count": original_sample_count,
                        "training_sample_count": sample_count,
                        "repeat_factor": int(config["repeat_factor"]),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )

    _write_control_state(output / "control.pt", raw_model.cpu().eval())


if __name__ == "__main__":
    main()
