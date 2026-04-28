#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <video|pure>" >&2
  exit 1
fi
shift || true

OPENFACE_BIN="/share1/zhaoqiqi/app/OpenFace/build/bin/FeatureExtraction"

case "${MODE}" in
  video)
    ## UBFC-rPPG
    # INPUT_ROOT="/share2/data/liangqian/UBFC_rPPG/dataset2"
    # OUTPUT_ROOT="/share1/zhaoqiqi/dataset/rPPG/processed/UBFC-rPPG"
    # PATTERN='vid.avi'

    ## CASME_2
    INPUT_ROOT="/share2/data/zhaoqiqi/dataset/rppg/CASME_2/rawvideo"
    OUTPUT_ROOT="/share2/data/zhaoqiqi/dataset/rppg/processed/CASME_2"
    PATTERN='*.avi'

    python script/extract_openface_landmarks.py \
      --mode video \
      --openface_bin "${OPENFACE_BIN}" \
      --input_root "${INPUT_ROOT}" \
      --output_root "${OUTPUT_ROOT}" \
      --pattern "${PATTERN}" \
      --recursive \
      --two_d_only \
      "$@"
    ;;
  pure)
    ## PURE
    INPUT_ROOT="/share2/data/liangqian/PURE"
    OUTPUT_ROOT="/share1/zhaoqiqi/dataset/rPPG/processed/PURE"
    
    python script/extract_openface_landmarks.py \
      --mode pure \
      --openface_bin "${OPENFACE_BIN}" \
      --input_root "${INPUT_ROOT}" \
      --output_root "${OUTPUT_ROOT}" \
      --two_d_only \
      # --skip_existing \
      "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use video or pure." >&2
    exit 1
    ;;
esac
