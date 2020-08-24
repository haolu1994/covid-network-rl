#!/bin/bash

dropbox=../../dropbox/phama.ai
data_name=uspto_multi
tpl_name=new
cooked_root=../cooked_data

save_dir=test_out

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python galixir_plan.py \
    -dropbox $dropbox \
    -model_for_test $dropbox/retrosyn_graph/model_dumps/uspto_multi.ckpt \
    -data_name $data_name \
    -save_dir $save_dir \
    -tpl_name $tpl_name \
    -cooked_root $cooked_root \
    -f_atoms $cooked_root/$data_name/atom_list.txt \
    -eval_normprob False \
    -gpu 0 \
    -topk 50 \
    -beam_size 50 \
    -target_mol $1 \
    $@


