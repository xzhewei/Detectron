export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
dataset=caltech
GPU=S1_G0-7
method_name=e2e_faster_rcnn_VGG-16-C5_10x

NUM_GPUS=8

output_dir=output/${dataset}/${GPU}-${method_name}/detectron-output
train_dir=${output_dir}/train/caltech_train_10x/generalized_rcnn

python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    NUM_GPUS $NUM_GPUS \
    EXP_ID 20190109D01