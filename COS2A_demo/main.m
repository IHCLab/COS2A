close all; clear;
addpath(genpath('./data')); 
addpath(genpath('./function'));
addpath(genpath('./DE_result'));

%% Setting
% rng(6);
normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
band_set = [23 12 5];

%% Load scene data
scene = 1; % Select the scene index (1:farm, ...)

fid = fopen('./data/data_list.txt');
scene_file = textscan(fid, '%s'); scene_file = scene_file{1};
fclose(fid);
scene_key = scene_file{scene};
load(fullfile('./data', [scene_key, '.mat']));

tic
%% Small Data Learning (DE) by COS2A
system(sprintf('conda run -n env python function/test_COS2A.py %d', scene));
load(fullfile('./DE_result', ['COS2A_result_' scene_key '.mat']));
Y_DE_COS2A = double(output);

%% CO-CNMF Algorithm (CO)
Y_S_10 = Y(:,:,[2,3,4,8]);
N = 10;
[D] = cal_D(Y_DE_COS2A, Y_S_10);
[Z_fused_COS2A, ~] = COCNMF_YDE_simp(Y_DE_COS2A, Y_S_10, N, D);
toc

tic
%% Small Data Learning (DE) by Universal Model
system(sprintf('conda run -n env python function/test_Universal.py %d', scene));
load(fullfile('./DE_result', ['Universal_result_' scene_key '.mat']));
Y_DE_Universal = double(output);

%% CO-CNMF Algorithm (CO)
Y_S_10 = Y(:,:,[2,3,4,8]);
N = 10;
[D] = cal_D(Y_DE_Universal, Y_S_10);
[Z_fused_Universal, ~] = COCNMF_YDE_simp(Y_DE_Universal, Y_S_10, N, D);
toc

%% Visualization
random_mode = 2; % 1: random pixels; 2: specified pixel coordinates
[random_indices] = plot_sample_curve(scene,random_mode, X, Y, Z_fused_COS2A, Z_fused_Universal);


figure('Position', [0 0 1200 400]);
subplot(1,4,1); imshow(normColor(X(:,:,band_set))); title('Y_{H} with Sample Pixels');  % AVIRIS
hold on; img_size = size(X); plot_img_with_sample(random_indices,img_size); hold off;
subplot(1,4,2); imshow(normColor(Y_S_10(:,:,[3,2,1]))); title('Y_{S}');                 % Sentinel-2
subplot(1,4,3); imshow(normColor(Z_fused_COS2A(:,:,band_set))); title('Y^{*}_{H} (COS2A)');  
subplot(1,4,4); imshow(normColor(Z_fused_Universal(:,:,band_set))); title('Y^{*}_{H} (Universal Model)');  

