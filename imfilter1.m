%load Indian_pines_corrected.mat£»

%load Indian_pines_corrected.mat£»
load Indian_pines_gt.mat;
data3D=indian_pines_corrected;
[h,w,b]=size(data3D);
img1=reshape(data3D,h*w,b);
 for i=1:size(data3D,3)
     data3D(:,:,i) = imfilter(data3D(:,:,i),fspecial('gaussian',1,5));
 end
 data3D = data3D./max(data3D(:));
 Data=data3D;   
 [h,w,b]=size(data3D);
img2=reshape(data3D,h*w,b);   
