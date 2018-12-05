dataset=scut
# GPU=0
MAX_ITER=220000

method_name=e2e-faster-rcnn-VGG16-C5-10x
output_dir=output/${dataset}/${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

rm $train_dir/model_final.pkl
python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    SOLVER.MAX_ITER $MAX_ITER
mv ${output_dir}/test ${output_dir}/iter$MAX_ITER

method_name=e2e-faster-rcnn-R50-C4-10x
output_dir=output/${dataset}/${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

rm $train_dir/model_final.pkl
python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    SOLVER.MAX_ITER $MAX_ITER
mv ${output_dir}/test ${output_dir}/iter$MAX_ITER

method_name=e2e-faster-rcnn-R101-C4-10x
output_dir=output/${dataset}/${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

rm $train_dir/model_final.pkl
python tools/train_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    OUTPUT_DIR $output_dir \
    SOLVER.MAX_ITER $MAX_ITER
mv ${output_dir}/test ${output_dir}/iter$MAX_ITER


