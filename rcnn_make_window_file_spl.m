function rcnn_make_window_file_spl(experiment_dir, subset, out_dir, ratio)
% rcnn_make_window_file(imdb, out_dir)
%   Makes a window file that can be used by the caffe WindowDataLayer 
%   for finetuning.
%
%   The window file format contains repeated blocks of:
%
%     # image_index 
%     img_path
%     channels 
%     height 
%     width
%     num_windows
%     class_index overlap x1 y1 x2 y2
%     <... num_windows-1 more windows follow ...>

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
imdb_train = imdb_from_common(experiment_dir, 'train');
imdb = imdb_from_common(experiment_dir, subset);
roidb = imdb.roidb_func(experiment_dir,imdb);

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

window_file = sprintf('%swindow_file_%s.txt', ...
    out_dir, imdb_train.name);
fid = fopen(window_file, 'a');

channels = 3; % three channel images
total_num = 0;
total_scores = [];
for i = 1:length(imdb.image_ids)
    roi = roidb.rois(i);
    I = find(~roi.gt & roi.boxes(:,5)>=0.7);
    total_num = total_num + length(I);
    total_scores = [total_scores;roi.boxes(I,5)];
end
select_num_boxes = floor(total_num*ratio);
[total_scores, ~] = sort(total_scores, 'descend');
if(select_num_boxes==0)
    min_score = inf;
else
    min_score = total_scores(select_num_boxes);
end


for i = 1:length(imdb.image_ids)
  tic_toc_print('make window file: %d/%d\n', i, length(imdb.image_ids));
  img_path = imdb.image_at(i);
  roi = roidb.rois(i);
  
  boxes = roi.boxes;
  I = find(~roi.gt & boxes(:,5)>=min_score);
  boxes = boxes(I,:);
  overlap = roi.overlap(I,:);
  weights = roi.weights(I);
  
  num_boxes = length(I);
%   total_num = size(boxes, 1);
  
  fprintf(fid, '# %d\n', i-1+length(imdb_train.image_ids));
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', ...
      channels, ...
      imdb.sizes(i, 1), ...
      imdb.sizes(i, 2));
  fprintf(fid, '%d\n', num_boxes);
  for j = 1:num_boxes
    [ov, label] = max(overlap(j,:));
    % zero overlap => label = 0 (background)
    if ov < 1e-5        % 1e-5
      label = 0;
      ov = 0;
    end
    bbox = boxes(j,:)-1;
    bbox = floor(bbox);
    if bbox(1)<0
        bbox(1) = 0;
    elseif bbox(1)>=imdb.sizes(i,2)
        bbox(1) = imdb.sizes(i,2)-1;
    end
    if bbox(2)<0
        bbox(2) = 0;
    elseif bbox(2) >= imdb.sizes(i,1)
            bbox(2) = imdb.sizes(i,1)-1;
    end
    if bbox(3)>=imdb.sizes(i,2)
        bbox(3) = imdb.sizes(i,2)-1;
    elseif bbox(3)<0
        bbox(3)=0;
    end
    if bbox(4)>=imdb.sizes(i,1)
        bbox(4) = imdb.sizes(i,1)-1;
    elseif bbox(4)<0
        bbox(4)=0;
    end
    
    
    fprintf(fid, '%d %.3f %d %d %d %d %.3f\n', ...
        label, ov, bbox(1), bbox(2), bbox(3), bbox(4),weights(j));
  end
end

fclose(fid);
