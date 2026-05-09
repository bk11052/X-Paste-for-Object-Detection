"""Distribution-aware Generative Copy-Paste augmentation package.

Submodules:
  distribution    - joint (class x scale x cx x cy) histogram + inverse-frequency sampler
  host_scene      - SegFormer-only host image analyzer with accept/reject criteria
  scale_heuristic - depth-free scale fallback (gt anchor -> cy regression -> distribution)
  style_match     - Lab histogram instance selection + post-paste L-channel matching
  build_augmented - top-level offline driver that produces an augmented YOLO split
  visualize       - bbox sanity visualization
"""

CLASSES = ("Soldier", "civilian_vehicle", "military_vehicle", "persons")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
