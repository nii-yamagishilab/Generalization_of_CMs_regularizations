#! /bin/bash

. ./path.sh

task='d1'
skip_test=true
start=1
end=1
ckpt=''

. ./utils/parse_options.sh

if [ $start -le -1 ] && [ $end -ge -1 ]; then
    ./local/prepare_data.sh ${task} ${skip_test}
fi

if [ $start -le 0 ] && [ $end -ge 0 ]; then
    echo "Compute duration..."
    wav-to-duration scp:data/${task}/train/wav.scp ark,t:data/${task}/train/utt2dur
    wav-to-duration scp:data/${task}/val/wav.scp ark,t:data/${task}/val/utt2dur
    if [ ! ${skip_test} = 'true' ]; then
        wav-to-duration scp:data/${task}/test/wav.scp ark,t:data/${task}/test/utt2dur
    fi
fi

if [ $start -le 1 ] && [ $end -ge 1 ]; then
    echo "Start training..."
    python train.py --mode train \
        --batch-size 200 \
        --task ${task} \
        --lr 0.001 \
        --momentum 0.9 \
        --weight-decay 0.0001 \
        --step-size 1 \
        --gamma 0.9
fi

if [ $start -le 2 ] && [ $end -ge 2 ]; then
    echo "Start evaluating..."
    python train.py --mode test --task ${task} --ckpt ${ckpt}
fi
