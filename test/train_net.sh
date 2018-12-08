export CUDA_VISIBLE_DEVICES=2
dataset=scut
GPU=S0_G2
method_name=e2e_frcnn_VGG16_C5d2_im1.5

MAX_ITER=180000

output_dir=output/${dataset}/${GPU}-${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    SOLVER.MAX_ITER $MAX_ITER