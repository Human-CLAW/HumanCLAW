from humanclaw_bench.batch import _child_process_env


def test_child_process_env_bounds_cpu_thread_pools() -> None:
    env = _child_process_env({"PATH": "/bin"})

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_child_process_env_preserves_explicit_overrides() -> None:
    env = _child_process_env(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "5",
        }
    )

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["NUMEXPR_NUM_THREADS"] == "5"
