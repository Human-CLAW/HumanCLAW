#!/usr/bin/env python3
"""Compatibility entry point for the production physics replay implementation.

Requires prepared HSSD and the release Habitat-Sim build. Example:

    PYTHONPATH=src python tests/runtime_forward_replay.py /path/to/rollout_00

The public experiment command accepts trace directories as well as old rollouts:
``humanclaw-bench replay /path/to/run --mode physics``.
"""

from humanclaw_bench.experiments.physics_replay import main, replay_and_compare


if __name__ == "__main__":
    main()
