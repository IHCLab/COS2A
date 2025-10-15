%close all; clear;
%load real_data
X = reshape(target,65536,172)';
[M L] = size(X);
d = mean(X,2);
U = X-d*ones(1,L);
[eV D] = eig(U*U');

for N=2:2:20
    C = eV(:,M-N+1:end);
    recons = C' * U;           
    Uhat = C * recons;          
    Xhat = Uhat + d*ones(1,L);
    ratio = (norm(Xhat, 'fro')^2) / (norm(X, 'fro')^2);

end

%Xhat2D=reshape(Xhat',256,256,172);
%figure;
%subplot(1,2,1);
%imshow(target(:,:,[30,20,10]))
%subplot(1,2,2);
%imshow(Xhat2D(:,:,[30,20,10]))
