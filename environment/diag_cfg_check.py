#!/usr/bin/env python3
"""Lightweight OpenPCDet-style _BASE_CONFIG_ merge + grid sanity check.
Does NOT need torch/pcdet; just validates the YAML chain & BEV grid math.
"""
import yaml
from pathlib import Path

ENV = Path(r"E:\Scholar\mmradarDetect\environment")


def winpath(p):
    # configs use WSL /mnt/e/... ; map to Windows for this checker
    s = str(p)
    if s.startswith("/mnt/e/"):
        return Path("E:/" + s[len("/mnt/e/"):])
    return Path(s)


def resolve_nested(cfg):
    """Recursively resolve any dict that itself carries a _BASE_CONFIG_."""
    if not isinstance(cfg, dict):
        return cfg
    if "_BASE_CONFIG_" in cfg:
        base = load_merged(cfg["_BASE_CONFIG_"])
        rest = {k: v for k, v in cfg.items() if k != "_BASE_CONFIG_"}
        cfg = deep_merge(base, {k: resolve_nested(v) for k, v in rest.items()})
        return cfg
    return {k: resolve_nested(v) for k, v in cfg.items()}


def load_merged(path):
    path = winpath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_BASE_CONFIG_", None)
    merged = {}
    if base:
        merged = load_merged(base)
    cfg = {k: resolve_nested(v) for k, v in cfg.items()}
    return deep_merge(merged, cfg)


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def grid_report(name, cfg):
    print(f"\n=== {name} ===")
    pcr = cfg.get("POINT_CLOUD_RANGE") or cfg.get("DATA_CONFIG", {}).get("POINT_CLOUD_RANGE")
    dp = (cfg.get("DATA_PROCESSOR") or cfg.get("DATA_CONFIG", {}).get("DATA_PROCESSOR") or [])
    voxel = None
    for step in dp:
        if step.get("NAME") == "transform_points_to_voxels":
            voxel = step["VOXEL_SIZE"]
    print("CLASS_NAMES:", cfg.get("CLASS_NAMES"))
    print("POINT_CLOUD_RANGE:", pcr)
    print("VOXEL_SIZE:", voxel)
    if pcr and voxel:
        gx = (pcr[3] - pcr[0]) / voxel[0]
        gy = (pcr[4] - pcr[1]) / voxel[1]
        print(f"BEV grid: {gx:.1f} x {gy:.1f}  (integer? {gx.is_integer() and gy.is_integer()})")
        # anchor head
        head = cfg.get("MODEL", {}).get("DENSE_HEAD", {})
        ag = head.get("ANCHOR_GENERATOR_CONFIG", [])
        for a in ag:
            stride = a.get("feature_map_stride")
            cell = voxel[0] * stride
            sz = a["anchor_sizes"][0]
            outx, outy = gx / stride, gy / stride
            print(f"  [{a['class_name']}] stride {stride} -> BEV cell {cell:.2f}m, "
                  f"out grid {outx:.0f}x{outy:.0f}, anchor {sz}, box/cell {sz[0]/cell:.2f}, "
                  f"matched {a['matched_threshold']}/{a['unmatched_threshold']}")


for stem in ["pointpillar_aiqii_full", "pointpillar_aiqii_smoke",
             "pointpillar_aiqii_multiclass_full", "pointpillar_mmaud_full"]:
    cfg = load_merged(ENV / "cfgs" / "pcdet_models" / f"{stem}.yaml")
    grid_report(stem, cfg)
