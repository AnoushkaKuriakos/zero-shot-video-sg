#!/usr/bin/env bash
# =============================================================================
#  Caption branch of the pipeline, step by step (run_all.sh calls these).
#  Outputs: $SG_OUT/<vid>/   Logs: ./logs/<out>_<vid>/<step>.log
#
#    bash run_pipeline.sh windows      import SAM3 masks, cut time windows   [$PY]
#    bash run_pipeline.sh dam          DAM description per tube              [$PY_DAM]
#    bash run_pipeline.sh ref          PixelRefer window captions            [$PY_REF]
#    bash run_pipeline.sh graph        extract + align triplets, fuse        [$PY]
#    bash run_pipeline.sh eval         Recall@K (only if the data has GT)    [$PY]
#    bash run_pipeline.sh video        video with tubes and relations        [$PY]
#
#  SMOKE=1 bash run_pipeline.sh windows dam ref graph  -> fake models, out_smoke/
#  Captioning resumes: delete captions_ref.jsonl to redo it.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ENVS=${ENVS:-$HOME/miniconda3/envs}
DATA=${DATA:?set DATA to the data root (tools/make_data_root.py)}
VIDEOS=${VIDEOS:-fam_1_mcs_2}
MASKS_FROM=${MASKS_FROM:?set MASKS_FROM to the SAM3 output dir}
export SG_OUT=${SG_OUT:-$PWD/out}
PY=${PY:-$ENVS/bakeoff/bin/python}      # Qwen2.5-VL (transformers 4.51)
PY_DAM=${PY_DAM:-$ENVS/dam/bin/python}  # describe-anything installed
PY_REF=${PY_REF:-$ENVS/vref/bin/python} # PixelRefer (transformers 4.46.3)
QWEN_ARGS=${QWEN_ARGS:---backend qwen}
DAM_ARGS=${DAM_ARGS:---dam-model nvidia/DAM-3B-Video}
DAM_REPO=${DAM_REPO:-}
PIXELREFER_ROOT=${PIXELREFER_ROOT:-$PWD/third_party/PixelRefer}
PIXELREFER_MODEL=${PIXELREFER_MODEL:-Alibaba-DAMO-Academy/PixelRefer-7B}
WIN_ARGS=${WIN_ARGS:-}           # e.g. "--win 30 --stride 15 --max-objects 10"
PARSE_ARGS=${PARSE_ARGS:-}       # e.g. --keep-all-predicates
CAP_ARGS=${CAP_ARGS:-}           # both captioners, e.g. "--prompt v3"
FUSE_ARGS=${FUSE_ARGS:-}         # e.g. --plausible

FAKE=()
if [ "${SMOKE:-0}" = 1 ]; then
  export SG_OUT=${SG_OUT%/out}/out_smoke; QWEN_ARGS="--backend fake"; FAKE=(--fake)
  echo ">> SMOKE: fake models, outputs in $SG_OUT"
fi
[ -d "$DATA/images" ] || { echo "DATA=$DATA has no images/ — export DATA=<sailvos3d root>"; exit 1; }
read -r -a V <<< "$VIDEOS"
A=(--data-root "$DATA" --videos "${V[@]}")
DAM_R=(); [ -n "$DAM_REPO" ] && DAM_R=(--dam-repo "$DAM_REPO")
LOGS=logs/$(basename "$SG_OUT")_${V[0]}${V[1]:+_etc}
mkdir -p "$LOGS"

run() {   # run <log name> <cmd...> : stream to terminal and append to the log
  echo "\$ ${*:2}" >> "$LOGS/$1.log"
  "${@:2}" 2>&1 | tee -a "$LOGS/$1.log"
}

step() {
  case "$1" in
  windows) run windows $PY make_windows.py "${A[@]}" --masks-from "$MASKS_FROM" $WIN_ARGS ;;
  dam)     run dam $PY_DAM describe_dam.py "${A[@]}" $DAM_ARGS "${DAM_R[@]}" "${FAKE[@]}" ;;
  ref)     run ref $PY_REF pixelrefer_caption.py "${A[@]}" --pixelrefer-root "$PIXELREFER_ROOT" \
               --model "$PIXELREFER_MODEL" "${FAKE[@]}" $CAP_ARGS ;;
  graph)   for P in ref; do
             if [ ! -f "$SG_OUT/${V[0]}/captions_$P.jsonl" ]; then echo "-- skip $P: no captions"; continue; fi
             run graph_$P $PY extract_relations.py "${A[@]}" --path $P $QWEN_ARGS
             run graph_$P $PY parse_relations.py   "${A[@]}" --path $P $QWEN_ARGS $PARSE_ARGS
             run graph_$P $PY fuse_relations.py    "${A[@]}" --path $P $FUSE_ARGS
           done ;;
  eval)    $PY tools/eval_pvsg.py "${A[@]}" --paths ref 2>&1 | tee "$LOGS/eval.log" ;;
  video)   run video $PY render_compare.py "${A[@]}" ;;
  all)     for s in windows dam ref graph video; do echo "=== $s"; step $s; done ;;
  *)       sed -n '2,15p' "$0"; exit 1 ;;
  esac
}
[ $# -gt 0 ] || { sed -n '2,15p' "$0"; exit 1; }
for s in "$@"; do step "$s"; done
