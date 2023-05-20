#! /bin/bash

score_file=$1

echo "Overall performance"
python ./compute_metric.py --score-filepath ${score_file}

root=`echo ${score_file} | cut -d'/' -f1,2`
mkdir -p ${root}/genres_score
for genre in drama entertainment interview live_broadcast movie play recitation singing speech vlog; do
    grep ${genre} ${score_file} > ${root}/genres_score/${genre}.score
    echo "${genre} performance"
    python ./compute_metric.py --score-filepath ${root}/genres_score/${genre}.score
done
