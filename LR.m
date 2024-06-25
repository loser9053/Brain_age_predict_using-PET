clear;
clc;
load('all.mat')
x_train=x_n(1:227,:);
y_train=y(1:227);
x_test=x_n(228:end,:);
y_test=y(228:end);
cvp = cvpartition(y_train,'k',10);
[fs,history] = sequentialfs(@class_LR,x_train,y_train,'cv',cvp,'direction','backward');

x_train=x_train(:,fs);
x_test=x_test(:,fs);


X_train=[ones(size(x_train,1),1),x_train];
[b,bint,r,rint,s]=regress(y_train,X_train);
predicted_train=b'*X_train';

X_test=[ones(size(x_test,1),1),x_test];
predicted_test=b'*X_test';

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
    X_test_new=[ones(size(x_test,1),1),x_test(randperm(size(x_test,1)),:)];
    pre_y_1=b'*X_test_new';
    correlation1(o)=corr(pre_y_1',y_test);    
end
p=mean(correlation1>r_test);