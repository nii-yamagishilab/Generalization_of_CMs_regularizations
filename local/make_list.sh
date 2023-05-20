#! /bin/bash

# for x in d1 d2 d3 d4; do
for x in d1; do
    cat ${x}/train.lst | rev | cut -d'/' -f3 | rev | sed -e 's:data:bonafide:g' > tmp.lst
    for y in `cat tmp.lst | sort | uniq`; do if [ ! ${y} = 'bonafide' ]; then sed -i "s:${y}:spoof:g" tmp.lst; fi; done
    paste -d' ' ${x}/train.lst tmp.lst > a && mv a ${x}/train.lst
    cat ${x}/val.lst | rev | cut -d'/' -f3 | rev | sed -e 's:data:bonafide:g' > tmp.lst
    for y in `cat tmp.lst | sort | uniq`; do if [ ! ${y} = 'bonafide' ]; then sed -i "s:${y}:spoof:g" tmp.lst; fi; done
    paste -d' ' ${x}/val.lst tmp.lst > a && mv a ${x}/val.lst
done
