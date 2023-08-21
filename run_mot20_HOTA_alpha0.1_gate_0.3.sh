# exp = 2023-07-02_02-37-43+best_paper_ablations_alpha0.1-gate0.3_post
python3 external/TrackEval/scripts/run_mot_challenge.py \
  --SPLIT_TO_EVAL val \
  --METRICS HOTA Identity CLEAR \
  --TRACKERS_TO_EVAL /remote-home/zhengyiyao/topictrack/results/trackers/MOT20-val/mot20_val_7_1/2023-07-01_15-06-46+best_paper_ablations_alpha0.0-gate0.3_post \
  --GT_FOLDER results/gt/ \
  --TRACKERS_FOLDER results/trackers/ \
  --BENCHMARK MOT20