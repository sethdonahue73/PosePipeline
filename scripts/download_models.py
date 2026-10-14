#!/usr/bin/env python
"""Pre-download PosePipe model weights into the locations the wrappers expect.

Mirrors the paths hard-coded in pose_pipeline/wrappers/{hand_estimation,mmpose,mmdet,hand_bbox,bridging}.py
so that DataJoint populate() calls find everything on disk and never hit the network.

Locations
  MODEL_DATA_DIR  = $PIPELINE_3RDPARTY, else <pose_pipeline pkg>/../3rdparty   (same rule as pose_pipeline/__init__.py)
  TFHUB_CACHE_DIR = MeTRAbs (Bridging_* top-down methods)
  TORCH_HOME      = torch hub cache, used only for the RTMDet hand-detector checkpoint

Set the same TFHUB_CACHE_DIR / TORCH_HOME / PIPELINE_3RDPARTY at pipeline run time, or the caches are not reused.

Usage (inside the project venv, so mim / mmpose / tensorflow_hub resolve):
  uv run python scripts/download_models.py --list
  uv run python scripts/download_models.py                     # route-b (default)
  uv run python scripts/download_models.py --groups hands topdown tracking
  uv run python scripts/download_models.py --groups all --root /data/posepipe_models
"""
import argparse
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------------------
# Model registry. Keys match the method names used in the wrappers / lookup tables.
# --------------------------------------------------------------------------------------

# mim-managed (mmpose 1.x / mmdet 3.x model index). dest = MODEL_DATA_DIR / subdir
MIM = {
    # HandPoseEstimationMethodLookup (wrappers/hand_estimation.py)
    "RTMPoseHand5": ("mmpose", "mmpose/RTMPoseHand5",
                     "rtmpose-m_8xb256-210e_hand5-256x256",
                     "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"),
    "RTMPoseCOCO": ("mmpose", "mmpose/RTMPoseCOCO",
                    "rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256",
                    "rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth"),
    "freihand": ("mmpose", "mmpose/freihand",
                 "td-hm_res50_8xb64-100e_freihand2d-224x224",
                 "res50_freihand_224x224-ff0799bc_20200914.pth"),
    "HRNet_dark": ("mmpose", "mmpose/HRNet_dark",
                   "td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256",
                   "hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth"),
    "HRNet_udp": ("mmpose", "mmpose/HRNet_udp",
                  "td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256",
                  "hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth"),
    # TopDownMethodLookup 15-17 (wrappers/mmpose.py)
    "RTMPose_coco-wholebody": ("mmpose", "mmpose/RTMPose_coco-wholebody",
                               "rtmpose-l_8xb32-270e_coco-wholebody-384x288",
                               "rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth"),
    "RTMPose_Cocktail14": ("mmpose", "mmpose/RTMPose_Cocktail14",
                           "rtmw-l_8xb320-270e_cocktail14-384x288",
                           "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"),
    "MMPose_VitPose_H": ("mmpose", "mmpose/MMPose_VitPose_H",
                         "td-hm_ViTPose-huge_8xb64-210e_coco-256x192",
                         "td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"),
    # tracking (wrappers/mmdet.py)
    "deepsort": ("mmdet", "mmdetection/deepsort",
                 "deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval",
                 "faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth"),
    "qdtrack": ("mmdet", "mmdetection/qdtrack",
                "qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval",
                "qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth"),
}

# Legacy mmpose 0.x checkpoints read from MODEL_DATA_DIR/mmpose/checkpoints (configs are committed).
# URLs follow the OpenMMLab 0.x model-zoo layout; a 404 is reported, not fatal.
_OMM = "https://download.openmmlab.com/mmpose/top_down"
LEGACY = {
    "HRNet_W48_COCO": f"{_OMM}/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth",
    "HRFormer_COCO": f"{_OMM}/hrformer/hrformer_base_coco_384x288-ecf0758d_20220316.pth",
    "HRNet_W48_COCOWholeBody": f"{_OMM}/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
    "HRNet_TCFormer_COCOWholeBody": f"{_OMM}/tcformer/tcformer_coco-wholebody_256x192-a0720efa_20220627.pth",
    "HRNet_W48_HALPE": f"{_OMM}/hrnet/hrnet_w48_halpe_384x288_dark_plus-d13c2588_20211021.pth",
}

# RTMDet hand detector (HandBboxMethodLookup 0). Config is not in the repo; checkpoint goes to torch hub cache.
RTMDET_HAND_CFG_URL = ("https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/"
                       "projects/rtmpose/rtmdet/hand/rtmdet_nano_320-8xb32_hand.py")
RTMDET_HAND_CFG_DEST = "mmpose/config/hand_2d_keypoint/rtmdet_nano_320-8xb32_hand.py"
RTMDET_HAND_CKPT_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                        "rtmdet_nano_8xb32-300e_hand-267f9c8f.pth")

METRABS_URL = "https://bit.ly/metrabs_l"  # Bridging_* (wrappers/bridging.py)

GROUPS = {
    "route-b": ["metrabs", "RTMPoseHand5"],           # TopDown 12 (Bridging_bml_movi_87) + HandPose 0
    "hands": ["RTMPoseHand5", "RTMPoseCOCO", "freihand", "HRNet_dark", "HRNet_udp"],
    "hand-bbox": ["rtmdet_hand"],
    "bridging": ["metrabs"],
    "topdown": ["RTMPose_coco-wholebody", "RTMPose_Cocktail14", "MMPose_VitPose_H"],
    "tracking": ["deepsort", "qdtrack"],
    "legacy": list(LEGACY),
}
GROUPS["all"] = sorted({m for g in GROUPS.values() for m in g})


# --------------------------------------------------------------------------------------
def default_root() -> Path:
    if os.environ.get("PIPELINE_3RDPARTY"):
        return Path(os.environ["PIPELINE_3RDPARTY"])
    # find_spec locates the package without executing it (importing pose_pipeline opens a DB connection)
    spec = importlib.util.find_spec("pose_pipeline")
    if spec is None or spec.origin is None:
        sys.exit("pose_pipeline not importable and PIPELINE_3RDPARTY unset; pass --root")
    return (Path(spec.origin).parent / ".." / "3rdparty").resolve()


def fetch(url: str, dest: Path, force: bool) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        print(f"  [get ] {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        print(f"  [ ok ] {dest}")
        return True
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"  [FAIL] {url}: {e}")
        return False


def get_mim(name: str, root: Path, force: bool) -> bool:
    from mim import download

    pkg, subdir, cfg_id, ckpt = MIM[name]
    dest = root / subdir
    cfg_path, ckpt_path = dest / f"{cfg_id}.py", dest / ckpt
    if cfg_path.exists() and ckpt_path.exists() and not force:
        print(f"  [skip] {dest}")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    try:
        download(pkg, [cfg_id], dest_root=str(dest))
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] mim download {pkg} {cfg_id}: {e}")
        return False
    ok = cfg_path.exists() and ckpt_path.exists()
    print(f"  [{' ok ' if ok else 'FAIL'}] {dest}"
          + ("" if ok else f"  (expected {cfg_path.name} and {ckpt_path.name}; got {sorted(p.name for p in dest.iterdir())})"))
    return ok


def get_metrabs(force: bool) -> bool:
    cache = os.environ.get("TFHUB_CACHE_DIR")
    if not cache:
        print("  [FAIL] TFHUB_CACHE_DIR unset; without it the pipeline cannot reuse the download")
        return False
    import tensorflow_hub as hub

    print(f"  [get ] {METRABS_URL} -> {cache}  (~2 GB)")
    try:
        path = hub.resolve(METRABS_URL)  # download + extract only; does not build the TF graph
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {e}")
        return False
    print(f"  [ ok ] {path}")
    return True


def get_rtmdet_hand(root: Path, force: bool) -> bool:
    ok = fetch(RTMDET_HAND_CFG_URL, root / RTMDET_HAND_CFG_DEST, force)
    from torch.hub import get_dir

    ckpt_dir = Path(get_dir()) / "checkpoints"  # where mmengine/torch.hub look for URL checkpoints
    return fetch(RTMDET_HAND_CKPT_URL, ckpt_dir / Path(RTMDET_HAND_CKPT_URL).name, force) and ok


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["route-b"], choices=sorted(GROUPS))
    ap.add_argument("--models", nargs="+", default=[], help="individual model keys, added to --groups")
    ap.add_argument("--root", type=Path, help="MODEL_DATA_DIR override (default: $PIPELINE_3RDPARTY or <pkg>/../3rdparty)")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--list", action="store_true", help="print groups and resolved paths, download nothing")
    args = ap.parse_args()

    root = (args.root or default_root()).resolve()
    names = list(dict.fromkeys([m for g in args.groups for m in GROUPS[g]] + args.models))

    print(f"MODEL_DATA_DIR = {root}")
    print(f"TFHUB_CACHE_DIR = {os.environ.get('TFHUB_CACHE_DIR', '(unset)')}")
    print(f"TORCH_HOME      = {os.environ.get('TORCH_HOME', '(unset -> ~/.cache/torch)')}")
    if root != Path(os.environ.get("PIPELINE_3RDPARTY", root)).resolve():
        print("WARNING: --root differs from $PIPELINE_3RDPARTY; the pipeline will look in the latter")

    if args.list:
        for g, ms in GROUPS.items():
            print(f"  {g:10s} {', '.join(ms)}")
        print(f"selected: {', '.join(names)}")
        return 0

    failed = []
    for name in names:
        print(f"\n== {name}")
        if name in MIM:
            ok = get_mim(name, root, args.force)
        elif name in LEGACY:
            ok = fetch(LEGACY[name], root / "mmpose/checkpoints" / Path(LEGACY[name]).name, args.force)
        elif name == "metrabs":
            ok = get_metrabs(args.force)
        elif name == "rtmdet_hand":
            ok = get_rtmdet_hand(root, args.force)
        else:
            print(f"  [FAIL] unknown model key {name!r}")
            ok = False
        if not ok:
            failed.append(name)

    print("\n" + ("all requested models present" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
