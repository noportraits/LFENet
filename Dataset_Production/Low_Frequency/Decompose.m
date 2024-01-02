clc;
clear all;
clear;
image_dir='D:/Desktop/zed/normal/low/right/';
image_files=dir([image_dir, '*.jpg']);
image_len = length(image_files);
disp(image_len);
for i=1:image_len

    img_name= [image_dir, image_files(i).name];
    img = im2double(imread(img_name));
    [m,n,c]=size(img);
    L = SideWindowBoxFilter(img, 5, 10);
    L_name=['D:/Desktop/zed/low_fre/low/right/', image_files(i).name(1:end-4),'.jpg'];
%    imwrite(L,L_name);
   disp(i)
end