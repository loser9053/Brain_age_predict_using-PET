clear;
clc;
load('all.mat')
x_train=x_n(1:227,:);
y_train=y(1:227);
x_test=x_n(228:end,:);
y_test=y(228:end);


bestcv=100;
indices=crossvalind('kfold',227,10);
j=1;
    for ridge_tmp=0.5:0.5:10
        predicted_cv_test=[];
        y_cv_test=[];
        for i=1:10
            test=(indices==i);
            train=~test;
            y_train_train=y_train(train);
            x_train_train=x_train(train,:);
            y_train_test=y_train(test);
            x_train_test=x_train(test,:);
            
            b=ridge(y_train_train,x_train_train,ridge_tmp);
            predicted_train_test=b'*x_train_test';
            predicted_cv_test=[predicted_cv_test;predicted_train_test'];
            y_cv_test=[y_cv_test;y_train_test];
        end
        rmse(j)=sqrt(mean((predicted_cv_test-y_cv_test).^2));
        
        if rmse(j)<bestcv
            bestcv=rmse(j);
            ridge_num=ridge_tmp;
        end
        j=j+1;
    end

save('best_ridge_num.mat','ridge_num');
cvp = cvpartition(y_train,'k',10);
[fs,history] = sequentialfs(@class_ridge,x_train,y_train,'cv',cvp,'direction','backward');

x_train=x_train(:,fs);
x_test=x_test(:,fs);


b=ridge(y_train,x_train,ridge_num);
predicted_train=b'*x_train';
predicted_test=b'*x_test';



mae_train=mean(abs(predicted_train'-y_train));
mse_train=mean((predicted_train'-y_train).^2);
nmse_train=mean((predicted_train'-y_train).^2./y_train.^2);   
r_train=corr(predicted_train',y_train);

delta=predicted_train'-y_train;
sigma=[ones(length(y_train),1),y_train];
B = regress(delta,sigma);
corrected_test_age=predicted_test'-B(2)*y_test-B(1);

mae_test=mean(abs(corrected_test_age-y_test));
mse_test=mean((corrected_test_age-y_test).^2);
nmse_test=mean((corrected_test_age-y_test).^2./y_test.^2);   
r_test=corr(corrected_test_age,y_test);

perm=1000;
for o=1:perm
    X_test_new=x_test(randperm(size(x_test,1)),:);
    pre_y_1=b'*X_test_new';
    correlation1(o)=corr(pre_y_1',y_test);    
end
p=mean(correlation1>r_test);