#!/usr/bin/env bash
# Whole pipeline on a data root made by tools/make_data_root.py:
#   SAM3 tubes -> windows -> caption branch (DAM, PixelRefer, triplets) -> direct branch (Qwen2.5-VL)
#   -> fused video scene graph -> classes corrected from DAM descriptions -> graph with object positions (stsg.json).
#
#   DATA=data/mysite VIDEOS="cam1 cam2" CLASSES="person,excavator,dump_truck:dump truck,ground" \
#     bash run_all.sh
# Optional: FILTER=--plausible (commonsense rules; written for the PSG4D-GTA classes only),
#           OUT=work (outputs go to $OUT/{sam3,caption,direct,graph}).
# Every GPU step runs under tools/guard.sh, one model at a time (needs ~48 GB GPU for crowded scenes).
set -euo pipefail
cd "$(dirname "$0")"
: "${DATA:?set DATA}"; : "${VIDEOS:?set VIDEOS}"; : "${CLASSES:?set CLASSES (printed by make_data_root.py)}"
OUT=${OUT:-$PWD/work}; FILTER=${FILTER:-}
ENVS=${ENVS:-$HOME/miniconda3/envs}; PY=${PY:-$ENVS/bakeoff/bin/python}
export PY PY_DAM=${PY_DAM:-$ENVS/dam/bin/python} PY_REF=${PY_REF:-$ENVS/vref/bin/python}
read -r -a V <<< "$VIDEOS"
g() { bash tools/guard.sh "$@"; }

# 1. segmentation and tracking
for v in "${V[@]}"; do
  [ -s "$OUT/sam3/$v/masks.txt" ] || g sam3_$v $PY tools/run_sam3.py --data-root "$DATA" --out "$OUT/sam3" \
      --videos "$v" --class-list "$CLASSES" --max-objects 10 --offload-video-cpu
done
# 2. caption branch (prompt v3, the final one)
for s in windows dam ref graph; do
  SG_OUT=$OUT/caption DATA=$DATA VIDEOS="$VIDEOS" MASKS_FROM=$OUT/sam3 CAP_ARGS="--prompt v3" \
    g caption_$s bash run_pipeline.sh $s
done
# 3. direct branch (reads the same windows)
mkdir -p "$OUT/direct"
for v in "${V[@]}"; do
  mkdir -p "$OUT/direct/$v"
  for f in masks.txt vocab.json windows.json dossiers.json; do cp -n "$OUT/caption/$v/$f" "$OUT/direct/$v/"; done
done
SG_OUT=$OUT/direct $PY som_render.py --data-root "$DATA" --videos "${V[@]}"     # numbered-mark keyframes
SG_OUT=$OUT/direct g direct $PY tools/vlm_direct.py --data-root "$DATA" --videos "${V[@]}" --backend qwen
# 4. fusion into one graph per video, then the exported graph
SG_OUT=$OUT/graph $PY tools/make_hybrid.py --a "$OUT/caption" --b "$OUT/direct" --videos "${V[@]}"
SG_OUT=$OUT/graph $PY fuse_relations.py --data-root "$DATA" --videos "${V[@]}" --path ref $FILTER
# 5. relabel each object tube from its DAM description (no merging); re-check the filter if it is on
SG_OUT=$OUT/graph $PY tools/relabel_dam.py --data-root "$DATA" --videos "${V[@]}" $FILTER
SG_OUT=$OUT/graph $PY tools/export_stsg.py --data-root "$DATA" --videos "${V[@]}"
echo "done: $OUT/graph/<video>/relations_ref.json (+ exported graph). Ask questions with:"
echo "  SG_OUT=$OUT/graph $PY tools/graph_rag.py --video ${V[0]} --captions $OUT/caption --questions my_questions.txt --backend qwen"
