G_logtostderr=1 ./external/caffe/build/tools/caffe train \
--solver=./code_human_detection/training_with_caffe/finetune_solver.prototxt \
--gpu=0 \
--weights=./code_human_detection/training_with_caffe/finetune_iter_500.caffemodel 2>&1 | tee -ia ./code_human_detection/training_with_caffe/log.txt


