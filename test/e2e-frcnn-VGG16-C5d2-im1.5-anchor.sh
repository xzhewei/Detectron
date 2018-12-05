dataset=scut
GPU=S2
method_name=e2e-frcnn-VGG16-C5d2-im1.5-anchor

output_dir=output/${dataset}/${GPU}-${method_name}/detectron-output
train_dir=${output_dir}/train/scut_train_10x/generalized_rcnn

MAX_ITER=180000

for((iter=19999;iter<$MAX_ITER;iter+=20000))
do
python tools/test_net.py \
    --cfg configs/${dataset}/${method_name}.yaml \
    TEST.WEIGHTS ${train_dir}/model_iter$iter.pkl \
    OUTPUT_DIR ${output_dir}/iter$iter
done

