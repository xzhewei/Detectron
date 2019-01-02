export CUDA_VISIBLE_DEVICES=4,5,6,7
dataset=scut
GPU=S1_G4-7
method_name=e2e_frcnn_VGG16-C5

NUM_GPUS=4

output_dir=output/${dataset}/${GPU}-${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    NUM_GPUS $NUM_GPUS \
    SOLVER.WEIGHT_DECAY 0.0001 \
    USE_NCCL True \
    EXP_ID 20181226D01