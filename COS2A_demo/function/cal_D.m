function [D] = cal_D(Y_DE, Y_S_10)
%-----------------------------------------------------------------------
% INPUT
%   Y_DE : the rough network solution (H x W x HSI band)
%   Y_S_10 : the multispectral image (H x W x MSI band)
% OUTPUT
%   D : the spectral response matrix (MSI band x HSI band)
%-----------------------------------------------------------------------
 
% Real Data with scene-adaptive D using ridge regression (11)   
eta = 0.0001;
[H, W, B] = size(Y_DE);
[h, w, b_10m] = size(Y_S_10);

Y_DE_2d = reshape(Y_DE, H*W, B)';
Y_S_10_2d = reshape(Y_S_10, h*w, b_10m)'; 

D = zeros(b_10m, B); % 4*172

for i = 1:b_10m
    Y_i = Y_S_10_2d(i, :);

    A_aug = [Y_DE_2d'; sqrt(eta) * eye(B)];
    b_aug = [Y_i' ; zeros(B, 1)];
    d_i = lsqnonneg(A_aug, b_aug);
    d_i = max(d_i, 0);
    D(i, :) = d_i';
end