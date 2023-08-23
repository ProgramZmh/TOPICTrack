exp=dance_val

python3 main.py --exp_name $exp --post --alpha_gate 0.9 --gate 0.2  --new_kf_off --aspect_ratio_thresh 1000 --w_assoc_emb 1.25

python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity CLEAR \
  --TRACKERS_TO_EVAL ${exp}_post \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK DANCE

python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity CLEAR \
  --TRACKERS_TO_EVAL ${exp} \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK DANCE
