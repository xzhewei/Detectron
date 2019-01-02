export CUDA_VISIBLE_DEVICES=8
dataset=scut
GPU=S1_G0-3
method_name=S1_e2e_frcnn_VGG16-C5d2-im1.5_roadline

NUM_GPUS=4

output_dir=output/${dataset}/${GPU}-${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    NUM_GPUS $NUM_GPUS \
    EXP_ID 20181226D01