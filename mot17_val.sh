exp=mot17_val_ablation
python3 main.py --exp_name $exp --post --alpha_gate 0.3 --gate 0.3 --new_kf_off --dataset mot17 --w_assoc_emb 0.75

python3 external/TrackEval/scripts/run_mot_challenge.py \
    --SPLIT_TO_EVAL val \
    --METRICS HOTA Identity CLEAR \
    --TRACKERS_TO_EVAL ${exp} \
    --GT_FOLDER results/gt/ \
    --TRACKERS_FOLDER results/trackers/ \
    --BENCHMARK MOT17

python3 external/TrackEval/scripts/run_mot_challenge.py \
    --SPLIT_TO_EVAL val \
    --METRICS HOTA Identity CLEAR \
    --TRACKERS_TO_EVAL ${exp}_post \
    --GT_FOLDER results/gt/ \
    --TRACKERS_FOLDER results/trackers/ \
    --BENCHMARK MOT17