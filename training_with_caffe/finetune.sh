G_logtostderr=1 ./external/caffe/build/tools/caffe train \
--solver=./code_human_detection/training_with_caffe/finetune_solver.prototxt \
--gpu=1 \
--weights=./data/caffe_nets/ilsvrc_2012_train_iter_310k 2>&1 | tee -i ./code_human_detection/training_with_caffe/log.txt


