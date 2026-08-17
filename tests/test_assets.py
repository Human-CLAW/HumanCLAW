from humanclaw_bench.assets import verify_bundled_assets


def test_all_bundled_assets_match_manifest():
    results = verify_bundled_assets()
    assert results
    assert all(row["ok"] for row in results), results
