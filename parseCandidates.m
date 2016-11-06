function [num_boxes, num_images] = parseCandidates( experiment_dir, subset, max_num_candidates )
%PARSECANDIDATES Summary of this function goes here
%   Detailed explanation goes here


nms_thres = 0.3;


dataset_dir = [experiment_dir 'dataset/'];

addpath([dataset_dir 'candidates_txt/']);
addpath([dataset_dir 'imagesets/']);

%%

num_boxes = 0;
num_images = 0;


image_list = textread([dataset_dir 'imagesets/' subset '.txt'],'%s');
num_image = length(image_list);

boxes = {};
images = {};

for i = 1:num_image
    images{1,i} = [image_list{i} '.png'];
    [x1,y1,x2,y2,score] = textread([dataset_dir 'candidates_txt/' subset '/' image_list{i} '.txt'],...
        '%d%d%d%d%f');
    boxes{1,i} = [y1,x1,y2,x2,score];

    [~,I] = sort(-score);
    boxes{1,i} = boxes{1,i}(I,:);
    score = score(I);

    boxes{1,i} = boxes{1,i}(1:min(max_num_candidates, end),:);
    score = score(1:min(max_num_candidates, end));


    num_boxes = num_boxes + size(boxes{i},1);
    num_images  = num_images + 1;

end

save([experiment_dir 'data/candidates/' subset '.mat'],'boxes','images');



end

