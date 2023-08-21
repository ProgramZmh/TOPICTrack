exp=baseline_mot20_test

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


python3 main.py --exp_name $exp --post --alpha_gate 0 --gate 0.3  --cmc_off --da_off --aw_off --grid_off --new_kf_off --dataset mot20 --test_dataset --w_assoc_emb 0.75 --aw_param 0.5