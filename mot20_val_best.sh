exp=mot20_val_ablations


python3 main.py --exp_name $exp --post --alpha_gate 0 --gate 0.3  --cmc_off --da_off --aw_off --grid_off --new_kf_off --dataset mot20 --w_assoc_emb 0.75 --aw_param 0.5

python3 external/TrackEval/scripts/run_mot_challenge.py \
        --SPLIT_TO_EVAL val \
        --METRICS HOTA Identity CLEAR \
        --TRACKERS_TO_EVAL ${exp} \
        --GT_FOLDER results/gt/ \
        --TRACKERS_FOLDER results/trackers/ \
        --BENCHMARK MOT20
      
python3 external/TrackEval/scripts/run_mot_challenge.py \
        --SPLIT_TO_EVAL val \
        --METRICS HOTA Identity CLEAR \
        --TRACKERS_TO_EVAL ${exp}_post \
        --GT_FOLDER results/gt/ \
        --TRACKERS_FOLDER results/trackers/ \
        --BENCHMARK MOT20




