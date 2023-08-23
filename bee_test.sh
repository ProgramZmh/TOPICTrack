
exp=bee_test
python3 main.py --exp_name $exp --post \
  --alpha_gate 0.9 --gate 0.099 --gate2 0.3 --iou_thresh 0.2 \
  --track_thresh 0.6 \
  --new_kf_off \
  --dataset bee \
  --aspect_ratio_thresh 1.6 --w_assoc_emb 0.7

python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity CLEAR \
  --TRACKERS_TO_EVAL ${exp} \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK BEE

python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity CLEAR \
  --TRACKERS_TO_EVAL ${exp}_post \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK BEE