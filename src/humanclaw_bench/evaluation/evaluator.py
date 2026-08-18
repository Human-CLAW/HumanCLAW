"""Run HumanClawBench episodes with three explicit output modes.

The default persists only per-call VLM JSON and the replay trajectory.  Video
and paper metrics are independent opt-in modes; their sensors and calculations
are not constructed unless the corresponding CLI flag is present.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from humanclaw_bench.agent.skills import skill_to_text
from humanclaw_bench.agent.types import PlannerResult
from humanclaw_bench.benchmark.episodes import HCFindNavInteractEpisode
from humanclaw_bench.config import ReleaseConfig
from humanclaw_bench.envs.find_nav_interact_env import HCFindNavInteractEnv
from humanclaw_bench.envs.half_physics_env import xb75_yup_to_half_physics_pose
from humanclaw_bench.evaluation.trajectory import (
    TrajectoryRecorder,
    build_replay_metadata,
)
from humanclaw_bench.paths import repository_root, resolve_release_path


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a JSON object from disk and reject non-object roots."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return resolved, value


def _load_model_config(profile: ReleaseConfig, path: str | Path) -> dict[str, Any]:
    """Validate the provider configuration required to call the VLM."""

    resolved, value = _load_json(path)
    contract = dict(profile.data.get("vlm") or {})
    required = tuple(
        contract.get(
            "required_model_config_fields",
            ("backend", "model", "max_tokens", "temperature", "response_format"),
        )
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(
            "Model config is incomplete; missing fields: " + ", ".join(missing)
        )
    if "temperature" in contract and float(value["temperature"]) != float(
        contract["temperature"]
    ):
        raise ValueError(
            f"Profile {profile.profile!r} requires temperature="
            f"{contract['temperature']}; model config requested {value['temperature']}"
        )
    if str(value.get("backend")) == "filesystem_queue":
        if not str(value.get("queue_dir") or ""):
            raise ValueError("filesystem_queue requires queue_dir")
        queue_dir = Path(str(value["queue_dir"])).expanduser()
        if not queue_dir.is_absolute():
            value["queue_dir"] = str((resolved.parent / queue_dir).resolve())
    return value


def _safe_name(value: str) -> str:
    """Sanitize an arbitrary value for use as one output-directory component."""

    characters = [
        char if (char.isalnum() or char in {"-", "_", "."}) else "_"
        for char in str(value)
    ]
    return "".join(characters).strip("_") or "unknown"


def _clear_generated_rollout_artifacts(output_dir: Path) -> None:
    """Remove only artifacts owned by an earlier incomplete attempt.

    An explicit rerun reuses the deterministic rollout directory. Without this
    bounded cleanup, a failed run that reached more decisions than the next
    run could leave stale step JSON beside the successful trajectory. The
    function never removes unknown user files or directories, and a fresh
    rollout pays only a few nonexistent-path checks.
    """

    fixed_names = (
        "ego.mp4",
        "exo.mp4",
        "metrics.json",
        "replay_manifest.json",
        "trajectory_before.npz",
        "trajectory_after.npz",
    )
    for name in fixed_names:
        (output_dir / name).unlink(missing_ok=True)
    for pattern in (
        "step???_percept_mid_low.json",
        "step???_verifier.json",
    ):
        for path in output_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()


def _write_step_vlm_records(
    output_dir: Path,
    step: int,
    decision: PlannerResult,
) -> list[Path]:
    """Write one compact record containing each logical stage's final attempt.

    Retry attempts remain in memory for exact call/token accounting, but only
    the terminal planner and verifier responses are useful as the readable
    episode log.  The normal no-retry path is unchanged.
    """

    suffixes = {
        "percept_mid_low": "percept_mid_low",
        "verifier": "verifier",
    }
    final_by_stage: dict[str, Any] = {}
    for stage in decision.stage_outputs:
        if not stage.prompt and not stage.raw_output and not stage.error:
            continue
        suffix = suffixes.get(stage.stage)
        if suffix is None:
            raise ValueError(f"Unsupported VLM stage: {stage.stage!r}")
        final_by_stage[stage.stage] = stage

    written: list[Path] = []
    for stage_name in suffixes:
        stage = final_by_stage.get(stage_name)
        if stage is None:
            continue
        suffix = suffixes[stage_name]
        payload: dict[str, Any] = {
            "prompt": stage.prompt,
            # On failure preserve the provider bytes semantically exactly,
            # including an empty string.  Replacing an empty response with an
            # empty parsed object would hide the distinction between "no
            # response" and a valid ``{}`` response.
            "response": stage.raw_output if stage.error else stage.raw,
        }
        if stage.error:
            payload["error"] = stage.error
        path = output_dir / f"step{step:03d}_{suffix}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def _history_item(step: int, decision: PlannerResult) -> dict[str, Any]:
    """Keep only the in-memory fields consumed by the next planner call."""

    raw_plan = decision.raw_plan
    return {
        "step": step,
        "action": decision.action.to_json(),
        "action_text": skill_to_text(decision.action),
        "visual_state_description": raw_plan.get("visual_state_description", ""),
        "reasoning_and_reflection": raw_plan.get("reasoning_and_reflection", ""),
        "current_subgoal": raw_plan.get("current_subgoal", ""),
        "language_plan": raw_plan.get("language_plan", ""),
        "planner_skill": decision.planner_skill,
        "verifier": decision.verifier,
    }


class HCFindNavInteractEvaluator:
    """Run one episode under the selected output-mode flags.

    Replay trajectories and per-call VLM JSON are the baseline contract.
    Video writers and the paper metric recorder are constructed only when the
    corresponding flags are true.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Store rollout options and load the selected self-contained release profile."""

        self.config = dict(config)
        self.profile = self.config.get("profile")
        self.output_root = Path(
            self.config.get("output_root") or repository_root() / "outputs"
        )
        self.env: Any = None
        self.motion: Any = None
        self.agent: Any = None
        self.seed_mode = ""
        self.seed_pkl: str | None = None
        self.seed_pt: str | None = None
        self._initial_xb_world_75: Any = None
        self._model_config: dict[str, Any] | None = None
        self.compute_metrics = bool(self.config.get("compute_metrics", False))
        self.save_video = bool(self.config.get("save_video", False))

    def check_config_valid(self) -> None:
        """Validate user options, model settings, and mode-specific output requirements."""

        if not isinstance(self.profile, ReleaseConfig):
            raise TypeError("config['profile'] must be a loaded ReleaseConfig")
        for key in ("model_config_path", "scene_id"):
            if not str(self.config.get(key) or "").strip():
                raise ValueError(f"Missing evaluation config value: {key}")
        benchmark = self.profile.section("benchmark")
        scene_override = self.config.get("scene_dataset_config")
        scene_config = (
            Path(scene_override).expanduser().resolve()
            if scene_override is not None
            else resolve_release_path(benchmark["scene_dataset_config"])
        )
        if not scene_config.is_file():
            raise FileNotFoundError(
                "Prepared HumanClaw HSSD scene config not found: "
                f"{scene_config}. Run `humanclaw-bench prepare-hssd "
                "--hssd-root /path/to/hssd-hab` first."
            )
        if int(self.config.get("n_rollouts", 1)) < 1:
            raise ValueError("n_rollouts must be positive")
        self._model_config = _load_model_config(
            self.profile,
            self.config["model_config_path"],
        )

    def evaluate_main(self) -> dict[str, Any]:
        """Construct episode, model, environment, motion runner, and ego agent, then evaluate."""

        if self._model_config is None:
            self.check_config_valid()
        assert isinstance(self.profile, ReleaseConfig)
        assert self._model_config is not None

        # Heavy model dependencies are imported only for rollout commands;
        # config, asset, metric-summary, and render commands stay lightweight.
        from humanclaw_bench.agent.ego_agent import PSVEgoAgent
        from humanclaw_bench.benchmark.episodes import (
            apply_instruction_version,
            load_episode,
        )
        from humanclaw_bench.motion.runner import MotionSkillRunner
        from humanclaw_bench.vlm.factory import build_model

        benchmark = self.profile.section("benchmark")
        agent_config = self.profile.section("agent")
        motion_config = self.profile.section("motion")
        physics = self.profile.section("physics")
        rendering = dict(self.profile.data.get("rendering") or {})
        max_steps = int(
            self.config.get("max_steps")
            if self.config.get("max_steps") is not None
            else benchmark["max_steps"]
        )
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        scene_override = self.config.get("scene_dataset_config")
        scene_config = (
            Path(scene_override).expanduser().resolve()
            if scene_override is not None
            else resolve_release_path(benchmark["scene_dataset_config"])
        )
        episode = load_episode(
            benchmark_dataset_dir=resolve_release_path(benchmark["dataset_dir"]),
            split=str(benchmark["split"]),
            scene_id=str(self.config["scene_id"]),
            scene_dataset_config=scene_config,
            episode_id=self.config.get("episode_id"),
            episode_index=int(self.config.get("episode_index", 0)),
            object_category=self.config.get("object_category"),
            max_steps=max_steps,
        )
        episode = apply_instruction_version(
            episode,
            str(benchmark["instruction_version"]),
        )

        output_override = self.config.get("output_root")
        if output_override is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_root = (
                repository_root()
                / "outputs"
                / (
                    f"{self.profile.profile}_{episode.scene_label}_ep{episode.episode_id}_"
                    f"{episode.object_category}_{stamp}"
                )
            )
        else:
            self.output_root = Path(output_override).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        model = build_model(self._model_config, self.output_root)
        # Asset/backend keys have explicit constructor arguments below.  Every
        # remaining physics key is a public environment setting, which keeps
        # the complete non-inheriting profile forwardable without duplication.
        physics_kwargs = {
            key: value
            for key, value in physics.items()
            if key not in {"backend", "agent_urdf", "agent_shift_npy", "physics_config"}
        }
        self.env = HCFindNavInteractEnv(
            scene_id=episode.scene_id,
            scene_dataset_config=episode.scene_dataset_config,
            half_physics_backend=str(physics["backend"]),
            agent_urdf=resolve_release_path(physics["agent_urdf"]),
            agent_shift_npy=resolve_release_path(physics["agent_shift_npy"]),
            physics_config=resolve_release_path(physics["physics_config"]),
            max_episode_steps=max_steps,
            lighting=str(rendering.get("lighting", "ambient")),
            ambient_strength=float(rendering.get("ambient_strength", 1.2)),
            ego_resolution=tuple(rendering.get("ego_resolution", [448, 448])),
            third_person_resolution=tuple(
                rendering.get("third_person_resolution", [512, 512])
            ),
            compute_metrics=self.compute_metrics,
            video_enabled=self.save_video,
            **physics_kwargs,
        )
        self.motion = MotionSkillRunner(
            skills=tuple(motion_config["skills"]),
            device=str(self.config.get("device") or motion_config["device"]),
            weights_root=str(motion_config["weights_root"]),
            weights_manifest=str(motion_config["weights_manifest"]),
            verify_weights=bool(motion_config["verify_weights"]),
        )
        self.agent = PSVEgoAgent(
            model,
            prompt_version=str(agent_config["prompt_version"]),
            verifier_version=str(agent_config["verifier_version"]),
            max_history=int(agent_config["max_history"]),
            plan_horizon_steps=int(agent_config["plan_horizon_steps"]),
        )
        self.seed_mode = str(motion_config["seed_mode"])
        self.seed_pkl = (
            str(resolve_release_path(motion_config["seed_pkl"]))
            if motion_config.get("seed_pkl")
            else None
        )
        self.seed_pt = (
            str(resolve_release_path(motion_config["seed_pt"]))
            if motion_config.get("seed_pt")
            else None
        )
        return self.run_episode_rollouts(
            episode,
            int(self.config.get("n_rollouts", 1)),
        )

    def rollout_dir(
        self, episode: HCFindNavInteractEpisode, rollout_index: int
    ) -> Path:
        """Return the deterministic output directory for one rollout index."""

        episode_dir = (
            f"{_safe_name(episode.scene_label)}_"
            f"ep{_safe_name(episode.episode_id)}_"
            f"{_safe_name(episode.object_category)}"
        )
        return self.output_root / episode_dir / f"rollout_{rollout_index:02d}"

    def _reset_env_for_rollout(self, episode: HCFindNavInteractEpisode) -> Any:
        """Reset motion and physics to the exact episode seed for an independent rollout."""

        init_xb_world = self.motion.reset(
            self.seed_mode,
            self.seed_pkl,
            self.seed_pt,
            episode.init_offset,
            episode.init_yaw,
        )
        self._initial_xb_world_75 = init_xb_world
        initial_transl, initial_orient, initial_body_pose = (
            xb75_yup_to_half_physics_pose(init_xb_world)
        )
        return self.env.reset(
            episode,
            initial_transl=initial_transl,
            initial_global_orient=initial_orient,
            initial_body_pose=initial_body_pose,
        )

    def _profile_name(self) -> str:
        """Return the profile label stored in replay metadata."""

        return self.profile.profile if isinstance(self.profile, ReleaseConfig) else ""

    def _new_trajectory_recorder(
        self,
        episode: HCFindNavInteractEpisode,
        rollout_index: int,
    ) -> TrajectoryRecorder:
        """Capture reset state once and create the always-on replay recorder."""

        initial_state = (
            self.env.replay_initial_state()
            if hasattr(self.env, "replay_initial_state")
            else {}
        )
        return TrajectoryRecorder(
            metadata=build_replay_metadata(
                profile_name=self._profile_name(),
                episode=episode,
                rollout_index=rollout_index,
                env=self.env,
            ),
            initial_xb_world_75=self._initial_xb_world_75,
            initial_state=initial_state,
        )

    def _new_metric_recorder(
        self,
        episode: HCFindNavInteractEpisode,
        rollout_index: int,
    ) -> Any:
        """Construct metrics lazily so normal rollout imports no metric stack."""

        if not self.compute_metrics:
            return None
        from humanclaw_bench.evaluation.metrics.episode import (
            PaperMetricRecorder,
        )

        metric_config = (
            dict(self.profile.data.get("metrics") or {})
            if isinstance(self.profile, ReleaseConfig)
            else {}
        )
        recorder = PaperMetricRecorder(
            episode=episode,
            env=self.env,
            config=metric_config,
            profile_name=self._profile_name(),
            rollout_index=rollout_index,
        )
        # This call occurs immediately after env.reset(), before the first
        # motion frame. It records the floor and initial-penetration exclusion
        # with one contact query; terminal pose restoration is unnecessary.
        recorder.record_reset()
        return recorder

    def _finalize_rollout_artifacts(
        self,
        *,
        output_dir: Path,
        trajectory: TrajectoryRecorder | None,
        metric_recorder: Any,
        video_writer: Any,
        rollout_succeeded: bool,
    ) -> None:
        """Close optional streams, write replay/metrics, then close Habitat.

        The nested ``finally`` blocks are deliberate.  A failed ffmpeg process
        must not leak Habitat, and a metric-finalization error must not prevent
        the compact replay bundle from being written for diagnosis.  Metrics
        are never emitted for a rollout whose main loop failed.
        """

        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            try:
                if trajectory is not None:
                    before, after = trajectory.materialize()
                    try:
                        if metric_recorder is not None and rollout_succeeded:
                            from humanclaw_bench.evaluation.metrics.episode import (
                                write_episode_metrics,
                            )

                            metrics = metric_recorder.finalize(
                                before=before,
                                after=after,
                            )
                            write_episode_metrics(
                                output_dir / "metrics.json",
                                metrics,
                            )
                    finally:
                        trajectory.write(output_dir)
            finally:
                self.env.close()

    def run_rollout(
        self, episode: HCFindNavInteractEpisode, rollout_index: int
    ) -> dict[str, Any]:
        """Execute one closed-loop episode and write only artifacts enabled by mode flags."""

        output_dir = self.rollout_dir(episode, rollout_index)
        output_dir.mkdir(parents=True, exist_ok=True)
        _clear_generated_rollout_artifacts(output_dir)

        history: list[dict[str, Any]] = []
        trajectory: TrajectoryRecorder | None = None
        metric_recorder: Any = None
        video_writer: Any = None
        rollout_succeeded = False

        try:
            if self.save_video:
                # Creating the exo sensor and ffmpeg pipes is opt-in; a normal
                # rollout never pays their render or I/O cost.
                from humanclaw_bench.evaluation.video import RolloutVideoWriter

                video_writer = RolloutVideoWriter(output_dir, float(self.env.fps))
                self.env.set_video_frame_sink(video_writer.append)

            self.agent.reset(episode)
            observation = self._reset_env_for_rollout(episode)
            if video_writer is not None:
                self.env.emit_initial_video_frame()
            trajectory = self._new_trajectory_recorder(
                episode,
                rollout_index,
            )
            metric_recorder = self._new_metric_recorder(
                episode,
                rollout_index,
            )

            for step in range(episode.max_steps):
                ego_image = Image.fromarray(observation.head_rgb)
                find_observation = (
                    self.env.metric_find_observation()
                    if metric_recorder is not None
                    else None
                )
                # Planner/verifier transport and JSON failures are handled
                # inside the agent at this unchanged simulator state.
                decision = self.agent.act(ego_image, history, [])
                # Each logical stage writes one small input/output JSON using
                # its final attempt.  There is deliberately no combined episode
                # log or second model-audit copy.
                _write_step_vlm_records(output_dir, step, decision)
                if metric_recorder is not None:
                    metric_recorder.record_decision(
                        step=step,
                        decision=decision,
                        find_observation=find_observation,
                    )

                action_text = skill_to_text(decision.action)
                if decision.action.skill == "stand":
                    # Stop/Stand commits the current realized pose; generating
                    # a redundant stand motion would change terminal metrics.
                    env_action: Any = {
                        "stop": True,
                        "skill": "stand",
                        "action": decision.action.to_json(),
                    }
                else:
                    # trajectory_before records exactly what Half-Physics will
                    # consume, before contacts and gravity change the pose.
                    generated_motion = self.motion.generate(
                        decision.action.skill,
                        decision.action.cond,
                    )
                    trajectory.record_before(
                        step=step,
                        action=decision.action,
                        action_text=action_text,
                        xb_world_75=generated_motion.xb_world_75,
                    )
                    env_action = generated_motion.xb_world_75

                observation, _reward, done, info = self.env.step(
                    env_action,
                    reasoning=decision.raw_plan,
                )
                body_state = info.get("body_state") if isinstance(info, dict) else None
                if isinstance(body_state, dict):
                    # trajectory_after is frame-aligned with the generated
                    # chunk and includes every dynamic object for fast replay.
                    assert trajectory is not None
                    trajectory.record_after(
                        step=step,
                        body_state=body_state,
                        object_states=dict(info.get("object_states") or {}),
                    )
                if metric_recorder is not None and decision.action.skill != "stand":
                    metric_recorder.record_motion(
                        step=step,
                        action_skill=decision.action.skill,
                        info=info if isinstance(info, dict) else {},
                    )
                history.append(_history_item(step, decision))
                if done:
                    break
            rollout_succeeded = True
        finally:
            # Motion generation is finished before artifact finalization.
            # Release its GPU tensors before trajectory compression and final
            # scalar aggregation so completed agents free batch GPU capacity.
            unload_motion = getattr(self.motion, "unload", None)
            if callable(unload_motion):
                unload_motion()
            self._finalize_rollout_artifacts(
                output_dir=output_dir,
                trajectory=trajectory,
                metric_recorder=metric_recorder,
                video_writer=video_writer,
                rollout_succeeded=rollout_succeeded,
            )

        return {
            "episode_id": episode.episode_id,
            "rollout_index": rollout_index,
            "steps": len(history),
            "episode_dir": str(output_dir),
            "metrics": (
                str(output_dir / "metrics.json") if self.compute_metrics else None
            ),
            "videos": (
                [str(output_dir / "ego.mp4"), str(output_dir / "exo.mp4")]
                if self.save_video
                else []
            ),
        }

    def run_episode_rollouts(
        self, episode: HCFindNavInteractEpisode, n_rollouts: int
    ) -> dict[str, Any]:
        """Run the requested independent rollouts for one prepared benchmark episode."""

        return {
            "rollouts": [
                self.run_rollout(episode, index) for index in range(n_rollouts)
            ]
        }


__all__ = ["HCFindNavInteractEvaluator", "_write_step_vlm_records"]
