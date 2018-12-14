rootpath=/home/xzw/code/Detectron/detectron/datasets/data
dataset=/home/all/datasets/SCUT_FIR_101

libpath=/home/all/code

cd $rootpath
rm -rf scut
mkdir scut
cd scut

ln -s $dataset/extract/test25/images ./scut_test
ln -s $dataset/extract/train01/images ./scut_train
ln -s $dataset/json ./json

rm -rf devkit
mkdir devkit
cd devkit
ln -s $libpath/datatool ./datatool
ln -s $libpath/toolbox ./toolbox