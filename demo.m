%% initialize
clear all,clc;

experiment_name = 'code_human_detection';
experiment_dir = [experiment_name '/'];


addpath(experiment_dir);
addpath([experiment_dir 'VOCcode/']);
addpath([experiment_dir 'imdb/']);


nms_thres = 0.3;
max_per_image =2000;




%% preprocessing
subsets = {'train', 'val', 'unlabel', 'test'};
max_num_candidates = [500,1000,2000,2000];
for i=1:length(subsets)
    subset = subsets{i};
    createxml(experiment_dir, subset);  
    [num_boxes, num_images] = parseCandidates(experiment_dir, subset, max_num_candidates(i));
    fprintf('%s num_images: %d    num_boxes:  %d    avg_num_boxes: %d\n',...
        subset, num_images, num_boxes, floor(num_boxes/num_images));

end


%% visualize gt bounding box

% subsets = {'test'};
% for i = 1:length(subsets)
%     subset = subsets{i};
%     img_dir = [experiment_dir 'dataset/images/'];
%     anno_dir = [ experiment_dir 'dataset/annotations/'];
%     images = textread([experiment_dir 'dataset/imagesets/' subset '.txt'], '%s');
% 
%     for i = 1:length(images)
%         img = imread([img_dir images{i} '.png']);
%         gt = VOCreadrecxml([anno_dir subset '/' images{i} '.xml']);
%         num = length(gt.objects);
%         boxes = [];
%         for j = 1:num
%             boxes(j,:) = gt.objects(j).bbox;
%         end
%         showboxes(img,boxes);
% 
%         pause;
%     end
% end


%% visualize candidates bounding box
% subsets = {'test'};
% for i = 1:length(subsets)
%     subset = subsets{i};
%     img_dir = [ experiment_dir 'dataset/images/'];
% 
%     img_ids = textread([experiment_dir 'dataset/imagesets/' subset '.txt'], '%s');
%     load([experiment_dir 'data/candidates/' subset '.mat']);
% 
%     for i = 1:length(img_ids)
%         img = imread([img_dir img_ids{i} '.png']);
%         box = boxes{i};
%         showboxes(img,box(1:min(20,end),[2 1 4 3]));
%         pause;
%     end
% end


%% create window file
fp_log = fopen([experiment_dir 'results/mAP.txt'],'w');
subsets = {'train', 'val', 'unlabel', 'test'};

for id=1:length(subsets)
    
    
    subset = subsets{id};
    imdb = imdb_from_common(experiment_dir, subset);
    if(strcmp(subset,'val') || strcmp(subset, 'test'))
        windowfile_outdir = [experiment_dir 'training_with_caffe/'];
        rcnn_make_window_file(experiment_dir,imdb, windowfile_outdir);
    end
    fprintf(fp_log, '%s    ', subset);
end
fprintf(fp_log, '\n');

%% self-paced learning
cache_name   = 'finetune_iter_500';
net_file = [windowfile_outdir cache_name '.caffemodel'];
crop_mode    = 'warp';
crop_padding = 16;


ratios = [0;0.1;0.2;0.4;0.6;0.9;1;1];
iterations = length(ratios);
aps_set = zeros(iterations,1);
    
st = 1;
for iter=st:iterations
    reset(gpuDevice(3));
    ratio = ratios(iter);
    windowfile_outdir = [experiment_dir 'training_with_caffe/'];
    imdb_train = imdb_from_common(experiment_dir, 'train');
    rcnn_make_window_file(experiment_dir, imdb_train, windowfile_outdir);
    
    % add unlabel confident data to train window file
    rcnn_make_window_file_spl(experiment_dir, 'unlabel', windowfile_outdir, ratio);
    
    % training cnn with caffe
    if(iter==1)
%         [status, results]=system(['./' windowfile_outdir 'finetune.sh'],'-echo');
    else
        [status, results]=system(['./' windowfile_outdir 'finetune_v2.sh'],'-echo');
    end
    
    cache_dir = [experiment_dir 'results/iteration_' num2str(iter) '/'];
    mkdir_if_missing(cache_dir);
    
    % forward cnn
   
    for id=1:length(subsets)
        subset = subsets{id};
        imdb = imdb_from_common(experiment_dir, subset);

        rcnn_exp_forward_common(imdb, ...
                experiment_dir, cache_dir, ...
                'crop_mode', crop_mode, ...
                'crop_padding', crop_padding, ...
                'net_file', net_file, ...
                'cache_name', cache_name);


        load([cache_dir 'rcnn_model.mat']);

        
        image_ids = imdb.image_ids;

        feat_opts = rcnn_model.training_opts;
        num_classes = length(rcnn_model.classes);


        aboxes = cell(num_classes, 1);
        box_inds = cell(num_classes, 1);
        for i = 1:num_classes
            aboxes{i} = cell(length(image_ids), 1);
            box_inds{i} = cell(length(image_ids), 1);
        end

        % heuristic that yields at most 100k pre-NMS boxes
        % per 2500 images
        % max_per_set = ceil(100000/2500)*length(image_ids);
        % max_per_image = 100;
        max_per_set = max_per_image*length(image_ids);


        top_scores = cell(num_classes, 1);
        thresh = -inf(num_classes, 1);
        box_counts = zeros(num_classes, 1);

        if ~isfield(rcnn_model, 'folds')
            folds{1} = 1:length(image_ids);
        else
            folds = rcnn_model.folds;
        end

        count = 0;
        if(strcmp(imdb.name, 'unlabel'))
            roidb = imdb.roidb_func(experiment_dir,imdb);
        end
        
        for f = 1:length(folds)
            for i = folds{f}
                count = count + 1;
                fprintf('%s: test (%s) %d/%d\n', procid(), imdb.name, count, length(image_ids));
                d = rcnn_load_cached_softmax_features(experiment_dir, feat_opts.cache_name, ...
                    imdb.name, image_ids{i});
                if isempty(d.feat)
                    continue;
                end
                
                zs = d.feat;
                
                if(strcmp(imdb.name, 'unlabel'))
                    new_boxes = [];
                    new_weights = [];
                end
                for j = 1:num_classes
                    boxes = d.boxes(:,1:4);
                    % here is j+1 not j, because class 1 means background     
                    z = zs(:,j+1);
                    
                    if(strcmp(imdb.name, 'unlabel')) 
                        new_boxes = [new_boxes;[boxes,z]];
                    end
                    I = find(~d.gt & z > thresh(j));
                    boxes = boxes(I,:);
                    scores = z(I);
                    aboxes{j}{i} = cat(2, single(boxes), single(scores));
                    [~, ord] = sort(scores, 'descend');
                    ord = ord(1:min(length(ord), max_per_image));
                    aboxes{j}{i} = aboxes{j}{i}(ord, :);
                    box_inds{j}{i} = I(ord);
                    
                    box_counts(j) = box_counts(j) + length(ord);
                    top_scores{j} = cat(1, top_scores{j}, scores(ord));
                    top_scores{j} = sort(top_scores{j}, 'descend');
                    if box_counts(j) > max_per_set
                      top_scores{j}(max_per_set+1:end) = [];
                      thresh(j) = top_scores{j}(end);
                    end
                end
                               
                if(strcmp(imdb.name, 'unlabel'))
                    scores = new_boxes(:,5);
                    [scores,I] = sort(scores,'descend');
                    roidb.rois(i).boxes = new_boxes(I,:);
                    roidb.rois(i).weights = calculate_weights(experiment_dir, roidb.rois(i).boxes(:,5));
                end
                
   
            end
        end
        for i = 1:num_classes
            % go back through and prune out detections below the found threshold
            for j = 1:length(image_ids)
                if ~isempty(aboxes{i}{j})
                    I = find(aboxes{i}{j}(:,end) < thresh(i));
                    aboxes{i}{j}(I,:) = [];
                    box_inds{i}{j}(I,:) = [];
                end
            end

            save_file = [cache_dir rcnn_model.classes{i} '_boxes_' imdb.name];
            boxes = aboxes{i};
            inds = box_inds{i};
            save(save_file, 'boxes', 'inds');
            clear boxes inds;
        end
        
        
        % Peform AP evaluation
        for model_ind = 1:num_classes
            cls = rcnn_model.classes{model_ind};
            res(model_ind) = imdb.eval_func(experiment_dir, cache_dir, cls, aboxes{model_ind}, imdb,'', nms_thres);
        end

        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('Results:\n');

        aps = [res(:).ap]';
        disp(aps);
        fprintf( '~~~~~~~~~~~~~~~~~~~~\n');
        aps_set(iter) = aps;
        fprintf(fp_log, '%f    ', aps);
      
        
        if(strcmp(imdb.name,'unlabel'))
            fprintf('Saving roidb to cache...');
            save([experiment_dir 'imdb/roidb_' imdb.name], 'roidb', '-v7.3');
            fprintf('done\n');
        end
    end
    fprintf(fp_log, '\n');
    
end

fclose(fp_log);
fprintf( '\n----------------------------\n');
fprintf('Total results:\n');
disp(aps_set);

%% visualize result bounding box
cache_dir = [experiment_dir 'results/iteration_1/'];
subsets = {'test'};
for i = 1:length(subsets)
    subset = subsets{i};
    img_dir = [experiment_dir 'dataset/images/'];

    img_ids = textread([experiment_dir 'dataset/imagesets/' subset '.txt'], '%s');
    load( [cache_dir 'res_boxes_test.mat']);
    load( [cache_dir 'thres_fppi1.mat']);

    for i = 1:length(img_ids)
        img = imread([img_dir img_ids{i} '.png']);
        img_height = size(img,1);
        img_width = size(img,2);

        top = floor(0.2 * img_height);
        left = floor(0.2 * img_width);

        box = res_boxes{i};
        idx = find(box(:,end)<thres_fppi1);
        box(idx,:) = [];
        box(:,[1,3]) = box(:,[1,3]) - left;
        box(:,[2,4]) = box(:,[2,4]) - top;
        w = box(:,3)-box(:,1);
        h = box(:,4)-box(:,2);
        box(:,1) = box(:,1) + w*12/60;
        box(:,3) = box(:,3) - w*12/60;
        box(:,2) = box(:,2) + h*12/120;
        box(:,4) = box(:,4) - h*12/120;
        img = img(top+1:img_height-top, left+1:img_width-left,:);
        disp(i);
        showboxes(img, box);
        pause(1);
    %     showboxes(img,box,[experiment_dir 'results/detection_results/', img_ids{i} '.png']);
    %     img = imread([experiment_dir 'results/detection_results/', img_ids{i} '.png']);
    %     img = img(2:end-1, 2:end-1,:);
    %     imwrite(img, [experiment_dir 'results/detection_results/', img_ids{i} '.png']);
    %    pause
    end

end


