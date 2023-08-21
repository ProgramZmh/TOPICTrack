# exp=best_paper_ablations
# python3 main.py --exp_name $exp --post --da_off --grid_off --new_kf_off --dataset mot20 --track_thresh 0.4 --w_assoc_emb 0.75 --aw_param 0.5


exp=mot20_val_ablations_no_two_round

folder_list=(
      "cache/embeddings/MOT20*"
      "cache/det_bytetrack_ablation.pkl"
      "results/trackers/MOT20-val/${exp}_post"
      "results/trackers/MOT20-val/${exp}"
    )

# 遍历文件夹列表
for folder_path in "${folder_list[@]}"; do
    rm -rf $folder_path
done


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




