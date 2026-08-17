<div align="right"><a href="README_zh.md">中文</a></div>

# HSSD val41 modifications

`humanclaw-hssd-val41/` contains every lightweight benchmark-specific config
and manifest:

- `scenes/`: 41 scene-instance JSON files. Their `object_instances` entries
  store template, transform, scale, and `STATIC`/`DYNAMIC` state.
- `objects/humanclaw/`: 14,537 per-instance physics object configs.
- `supplement.json`: immutable HF location, archive checksum, and per-file
  records for 1,693 instance-specific baked meshes absent from official HSSD.
- `hssd-hab.scene_dataset_config.json`: the Habitat-Sim dataset entry point.
- `asset_requirements.json`: names, sizes, and hashes of every official and
  supplemental asset.

`humanclaw-bench prepare-hssd` downloads the pinned supplement from the gated
`HumanCLAW/HumanCLAW-HSSD` dataset when it is not cached, verifies the archive
and every GLB, copies the explicit JSON files, and links matching official and
supplemental meshes, stages, and semantics. The supplement preserves baked
instance scales and exact render-mesh colliders; the command fails instead of
substituting a coarse official collider when an expected file or hash is
missing. Pass a local archive or extracted directory with `--supplement` on an
offline machine.

Scene `104862384_172226319` contains one accessibility correction. Its
non-openable shower enclosure (object instance 166) is placed below the active
scene so agents starting in the bathroom are not trapped inside it. The entry
remains at index 166 solely to preserve every later Habitat object ID; the
shower is absent from both rendering and collision for all episodes in this
scene.
