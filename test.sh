dir=$(ls -l /remote-home/zhengyiyao/topictrack/results/trackers/MOT20-val/mot20_val_7_1/ |awk '/^d/ {print $NF}')
str='post'
for i in $dir
do
    
    result=$(echo $i | grep "${str}")
    if [ "$result" != "" ]
    then
        echo $i
        python3 external/TrackEval/scripts/run_mot_challenge.py \
            --SPLIT_TO_EVAL val \
            --METRICS HOTA Identity CLEAR \
            --TRACKERS_TO_EVAL /remote-home/zhengyiyao/topictrack/results/trackers/MOT20-val/mot20_val_7_1/$i \
            --GT_FOLDER results/gt/ \
            --TRACKERS_FOLDER results/trackers/ \
            --BENCHMARK MOT20

    else
        echo "no"
    fi  
done   