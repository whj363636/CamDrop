#!/bin/bash
set +x
export CUDA_VISIBLE_DEVICES=${1}

ak_array=('PGD')

for ak_type in ${ak_array[@]}
do 
  echo ${ak_type}
  ./attack.py \
  --data "/home/wangguangrun/ILSVRC2012/" \
  --load ${2}\
  --norm ${3} \
  --batch ${4} \
  --eval \
  --ak_type ${ak_type}
done