addpath('./utils');
load('SomeISP_operator_python/201_CRF_data.mat');
load('SomeISP_operator_python/dorfCurvesInv.mat');
s_path_left = 'D:/mydataset/bestbignoise/kitti/test/normal/gt/left';
s_path_right = 'D:/mydataset/bestbignoise/kitti/test/normal/gt/right';

source = [s_path_left,'/*.png'];
namelist = dir(source);
outpath_left = 'D:/mydataset/bestbignoise/kitti/test/normal/low/left';
outpath_right = 'D:/mydataset/bestbignoise/kitti/test/normal/low/right';
len = length(namelist);
disp(len);
l = 0;
for i=1 : len
    disp(i);
    I_gl = I;
    B_gl = B;
    I_inv_gl = invI;
    B_inv_gl = invB;
    img_path = [s_path_left,'/', namelist(i).name];
    Img1 = imread(img_path);
    disp(img_path);
    Img1 = im2single(Img1);
    s_a = 0.09; % 空间噪声的下限
    s_b = 0.1; % 空间噪声的上限
    c_a = 0.006; % 色彩噪声的下限
    c_b = 0.007; % 色彩噪声的上限
    sigma_s = rand(1, 3) * (s_b - s_a) + s_a; % recommend 0~0.16 空间噪声
    sigma_c = rand(1, 3) * (c_b - c_a) + c_a; % recommend 0~0.06 色彩噪声
%     sigma_s = [0,0,0];
%     sigma_c = [0,0,0];
    CRF_index = 5;  % 1~201
    pattern = 1;    % 1: 'gbrg', 2: 'grbg', 3: 'bggr', 4: 'rggb', 5: no mosaic
    noise1 = AddNoiseMosai(Img1,I_gl,B_gl,I_inv_gl,B_inv_gl, sigma_s, ...
        sigma_c, CRF_index, pattern);
    qality = 70; % image quality, recommend 60~100
    str = [outpath_left,'/',  namelist(i).name];
    disp(str);
%     imwrite(noise1, str); 
    
    
    
    name = strrep(namelist(i).name, "left", "right");
    p = strcat(s_path_right, '/', name);
    p = char(p);
    disp(p);
    Img2 = imread(p);
    Img2 = im2single(Img2);
%     sigma_s = [0,0,0];
%     sigma_c = [0,0,0]; 
    CRF_index = 5;  % 1~201
    pattern = 1;    % 1: 'gbrg', 2: 'grbg', 3: 'bggr', 4: 'rggb', 5: no mosaic
    noise2 = AddNoiseMosai(Img2,I_gl,B_gl,I_inv_gl,B_inv_gl, sigma_s, ...
        sigma_c, CRF_index, pattern);
    out = strcat(outpath_right, '/', name);
    out = char(out);
    disp(out);
%     imwrite(noise2, out);
end
% imshow(cat(2, Img, noise), 'InitialMagnification', 'fit');