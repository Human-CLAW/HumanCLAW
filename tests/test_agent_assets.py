"""Validate the paper-default and optional finger-separated humanoids."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import pytest

from humanclaw_bench.assets import agent_asset_registry, resolve_agent_asset
from humanclaw_bench.config import load_config
from humanclaw_bench.paths import repository_root


FINGER_NAMES = tuple(
    f"{side}_{finger}{joint}"
    for side in ("left", "right")
    for finger in ("index", "middle", "pinky", "ring", "thumb")
    for joint in (1, 2, 3)
)


def test_optional_agent_registry_preserves_the_paper_default() -> None:
    """Keep the reported configuration pinned to the hand-merged humanoid."""

    registry = agent_asset_registry()
    assert registry["default"] == "paper"
    assert set(registry["assets"]) == {"paper", "finger-separated"}
    paper_urdf, paper_shift = resolve_agent_asset("paper")
    profile = load_config("paper_fullval_v1")
    assert paper_urdf == profile.path_value("physics", "agent_urdf")
    assert paper_shift == profile.path_value("physics", "agent_shift_npy")
    assert registry["assets"]["finger-separated"]["paper_default"] is False


def test_finger_separated_urdf_is_self_contained_and_independent() -> None:
    """Require visual and collision geometry on all thirty finger links."""

    urdf, shift = resolve_agent_asset("finger-separated")
    assert shift.is_file()
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == (
        "caf3be74ee250469b20a5e7fcb18a2f0959854a6af1e337ba00f8d85f559059d"
    )
    root = ET.parse(urdf).getroot()
    links = {element.attrib["name"]: element for element in root.findall("link")}
    assert len(links) == 55
    assert len(root.findall("joint")) == 54
    for name in FINGER_NAMES:
        link = links[name]
        visual = link.find("visual/geometry/mesh")
        collision = link.find("collision/geometry/mesh")
        assert visual is not None, name
        assert collision is not None, name
        for mesh in (visual, collision):
            path = urdf.parent / mesh.attrib["filename"]
            assert path.is_file(), path


def test_unknown_agent_asset_is_rejected() -> None:
    """Fail before simulator startup when an optional asset name is invalid."""

    with pytest.raises(ValueError, match="Unknown agent asset"):
        resolve_agent_asset("unknown")


def test_agent_registry_contains_only_release_relative_paths() -> None:
    """Prevent machine-local asset locations from entering the public registry."""

    root = repository_root()
    for entry in agent_asset_registry()["assets"].values():
        for key in ("urdf", "shift"):
            assert not str(entry[key]).startswith("/")
            assert (root / entry[key]).is_file()
