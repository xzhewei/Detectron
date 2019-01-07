export CUDA_VISIBLE_DEVICES=0,1
dataset=scut
GPU=S0_G01
method_name=S0_e2e_frcnn_VGG16-C5_roadline

NUM_GPUS=2

output_dir=output/${dataset}/${GPU}-${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    NUM_GPUS $NUM_GPUS \
    EXP_ID 20181220D01