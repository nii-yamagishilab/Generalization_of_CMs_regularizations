#! /bin/bash

task=$1
skip_test=$2
if [ ${skip_test} = 'true' ]; then
    processing='train val'
else
    processing='train val test'
fi
for x in `echo ${processing} | cut -d' ' -f1-`; do
    echo "Preparing ${x}..."
    cat data/${task}/${x}/wav.lst | rev | cut -d'/' -f3 | rev | sed -e 's:data:real:g' > data/${task}/${x}/tmp.lst
    for y in `cat data/${task}/${x}/tmp.lst | sort | uniq`; do if [ ! ${y} = 'real' ]; then sed -i "s:${y}:spoof:g" data/${task}/${x}/tmp.lst; fi; done
    cat data/${task}/${x}/wav.lst | rev | cut -d'/' -f1,2,3 | rev | sed -e 's:/:-:g' | paste -d'-' data/${task}/${x}/tmp.lst - > data/${task}/${x}/uttid
    paste -d' ' data/${task}/${x}/uttid data/${task}/${x}/wav.lst | sort > data/${task}/${x}/wav.scp
    cat data/${task}/${x}/uttid | sort > data/${task}/${x}/tmp && mv data/${task}/${x}/tmp data/${task}/${x}/uttid
    cut -d'-' -f1 data/${task}/${x}/uttid > data/${task}/${x}/tmp.lst
    paste -d' ' data/${task}/${x}/uttid data/${task}/${x}/tmp.lst > data/${task}/${x}/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${task}/${x}/utt2spk > data/${task}/${x}/spk2utt
    cat data/${task}/${x}/wav.scp | rev | cut -d'/' -f1 | rev | cut -d'-' -f1 | paste -d' ' data/${task}/${x}/uttid - > data/${task}/${x}/utt2genre
    rm data/${task}/${x}/uttid data/${task}/${x}/tmp.lst
    echo "${x} done!"
done
