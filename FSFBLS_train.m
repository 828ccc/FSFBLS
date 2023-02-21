clear all;close all;clc
warning off all;
format compact;
%% Load data from the indian pines database
tic
OA=zeros(16,10);
z0=0;
%  blsOA=zeros(10,1);
%  EPFAA=zeros(10,1);
%  EPFCA=zeros(10,16);
%  EPFKA=zeros(10,1);
% for N1=50:10:80
%     for N2=10:10:100
%         for N3=10:10:100
                        
%for para=1:4
  %for N1=10:2:70
%     for N2=10:10:100
%         for N3=500:50:1000
%           for para=2:2:10
%               for t=1:1:10
                for times=1:1
                   load Indian_pines_corrected.mat;
                   load Indian_pines_gt.mat;
                   label_gt =indian_pines_gt;
                   global indian_pines_gaborall;
                   data3D=indian_pines_corrected;
                   
                   
                  
                  %%%%%%%%Gabor滤波器%%%%%%%%%%%
%                       for dir=1:10
%                         indian_pines_gaborall(1:145,1:145,1:200,dir)=G_I(dir); %indian_pines_gaborall存储所有的gabor特征，数据结构为（x,y,b,dir）                          
%                         fprintf('band %2.0f is completed!\n',dir);
%                       end 
%                       save('indian_pines_gaborall_1.mat','indian_pines_gaborall');
%                       data3D=indian_pines_gaborall;
                      %data3D=salinas_corrected;
                     % Data=data3D;
                      %data3D=indian_pines_corrected;
                      %%%%%把滤波之后的矩阵和原始矩阵合并起来
                      
%                       
%                       [h,w,b]=size(indian_pines_gaborall);
%                       data3D=reshape(indian_pines_gaborall,h*w,b);
%                       Data=data3D;
                      
                      
                      
%                       img1=[indian_pines_gaborall;indian_pines_corrected];
%                       C = cat(3, img1, indian_pines_corrected);
%                       [h,w,b]=size(indian_pines_gaborall);
%                       img1=reshape(indian_pines_gaborall,h*w,b);
%                       [H,W,B]=size(hsi_data_original);
%                       img2=reshape(hsi_data_original,H*W,B);
%                       img3=[img1,img2]
%                       data3D=img3;
%                       Data=data3D;
                     
% % 
                      %高斯滤波
                      para=4;
                      t=4;
                       for i=1:size(data3D,3)
                            data3D(:,:,i) = imfilter(data3D(:,:,i),fspecial('gaussian',para,t));
                        end
                        data3D = data3D./max(data3D(:));
                        Data=data3D;   
%                       
%                       [a,b,c]=size(Data);
%                       Data=reshape(Data,a*b,c);
%                       A=Data(1:32,1:32); 
%                       I=mat2gray(A);
%                       figure,imshow(I);
%                       
                      
                     
                       
                       
                       
                       
                       
                        trainpercentage=5;
                        [DataTest DataTrain  TrainLabel TestLabel testindex trainindex] = samplesdivide_5(Data,label_gt, trainpercentage);
                        trainindexrand=trainindex;
                        testindexrand=testindex;                  
                        [H, W, B] = size(data3D);
                        img1=reshape(data3D,[H*W,B]);                        
                        max_d = max(img1(:));
                        min_d = min(img1(:));
                        img1 = (img1-min_d)/(max_d-min_d);
                        data3D=img1;                        
                        DataTrain=data3D(trainindexrand(:), :);
                        DataTest=data3D(testindexrand, :);
                        train_x=DataTrain;
                        test_x=DataTest;
                        maplable=reshape(label_gt,H*W,1);
                        trainlabel=maplable(trainindexrand(:), :);
                        testlabel=maplable(testindexrand, :);
                        trainlen=length( trainlabel);
                        testlen=length(testlabel);
                        train_y=zeros(trainlen,16);
                        test_y=zeros(testlen,16);

                        Trainkey=unique(trainlabel);
                        for i=1:length(Trainkey)
                            for j =1:trainlen
                                if trainlabel(j,1)==Trainkey(i,1)
                                    train_y(j,i)=1;
                                end    
                            end

                        end

                        Testkey=unique(testlabel);
                        for i=1:length(Testkey)
                            for j =1:testlen
                                if testlabel(j,1)==Testkey(i,1)
                                    test_y(j,i)=1;
                                end    
                            end

                        end


                        train_y=(train_y-1)*2+1;
                        test_y=(test_y-1)*2+1;
                        assert(isfloat(train_x), 'train_x must be a float');
                        assert(all(train_x(:)>=0) && all(train_x(:)<=1), 'all data in train_x must be in [0:1]');
                        assert(isfloat(test_x), 'test_x must be a float');
                        assert(all(test_x(:)>=0) && all(test_x(:)<=1), 'all data in test_x must be in [0:1]');
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                        %%%%%%%%%%%%%%%%%%%%This is the model of broad learning sytem with%%%%%%
                        %%%%%%%%%%%%%%%%%%%%one ot structrue%%%%%%%%%%%%%%%%%%%%%%%%
                       C = 2^-10;   %----C: the regularization parameter for sparse regualarization
                       s = .9;          %----s: 增强节点的收缩参数
                       best = 0.62;
                       result = [];
                       for NumRule=70 %1:1:20                 %每个模糊子系统的模糊规则搜索范围
                          for NumFuzz=80 %1:1:20              %模糊子系统个数搜索范围
                             for NumEnhan=70  %1:1:20      %增强节点搜索范围
                                clc;
                                rand('state',1);
                                for i=1:NumFuzz
                                   alpha=rand(size(train_x,2),NumRule);
                                   Alpha{i}=alpha;
                                 end  %generating coefficients of the then part of fuzzy rules for each fuzzy system
                                 WeightEnhan=rand(NumFuzz*NumRule+1,NumEnhan); %%Iinitializing  weights connecting fuzzy subsystems  with enhancement layer
            
                                 fprintf(1, 'Fuzzy rule No.= %d, Fuzzy system. No. =%d, Enhan. No. = %d\n', NumRule, NumFuzz,NumEnhan);
                                 [NetoutTest,Training_time,Testing_time,TrainingAccuracy,TestingAccuracy,preditlabel]  = fbls_train(train_x,train_y,test_x,test_y,Alpha,WeightEnhan,s,C,NumRule,NumFuzz);
                                 time =Training_time + Testing_time;
                                 result = [result; NumRule NumFuzz NumEnhan TrainingAccuracy TestingAccuracy];
                                 if best < TestingAccuracy
                                 best = TestingAccuracy;
                                 save optimal.mat TrainingAccuracy TestingAccuracy  NumRule NumFuzz NumEnhan time
                                 end
                                 % clearvars -except best NumRule NumFuzz NumEnhan train_data train_label test_data test_label  s C result NetoutTest
                              end
                            end
                       end
                       %[EPFOA,AA,kappa,CA]=confusion(testlabel,preditlabel);
                


                         blsOA(times,1)=TestingAccuracy;
                         predlabel=preditlabel;
                       
                        testlen=length(predlabel);
                        BLSresult_Indian=maplable;
                        for i=1:testlen
                            k=testindexrand(i,1);
                            BLSresult_Indian(k,1)=predlabel(i,1);
                        end

                        BLSresult_Indian=reshape(BLSresult_Indian,H,W);
                      
                        EPFresult_Indian = EPF(3,1,img1,BLSresult_Indian);
                %         toc
                        
                        EPFresult_Indian =reshape(EPFresult_Indian,[H*W 1]);
                        
                        EPFresulttest=EPFresult_Indian(testindexrand,1) ;
                        %%%% Evaluation the performance of the EPF
                        [EPFOA,AA,kappa,CA]=confusion(testlabel,EPFresulttest);
%                         
                        OA(para,times)=EPFOA;

                end
%                 par=[N1 80 70 EPFOA * 100];
%                      disp(par);
%                        z0=z0+1;
%                        AAAEPFOA_NumEnhan(z0,:)=par;              
%         end
%      end

