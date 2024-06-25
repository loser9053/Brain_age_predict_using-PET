clear;
clc;
load('all.mat')
x_train=x_n(1:227,:);
y_train=y(1:227);
x_test=x_n(228:end,:);
y_test=y(228:end);
cvp = cvpartition(y_train,'k',10);
[fs,history] = sequentialfs(@class_RT,x_train,y_train,'cv',cvp,'direction','backward');

x_train=x_train(:,fs);
x_test=x_test(:,fs);


model=RegressionTree.fit(x_train,y_train);
predicted_train=predict(model,x_train);
predicted_test=predict(model,x_test);

mae_train=mean(abs(predicted_train-y_train));
mse_train=mean((predicted_train-y_train).^2);
nmse_train=mean((predicted_train-y_train).^2./y_train.^2);   
r_train=corr(predicted_train,y_train);

mae_test=mean(abs(predicted_test-y_test));
mse_test=mean((predicted_test-y_test).^2);
nmse_test=mean((predicted_test-y_test).^2./y_test.^2);   
r_test=corr(predicted_test,y_test);

perm=1000;
for o=1:perm
    x_test_new=x_test(randperm(size(x_test,1)),:);
    pre_y_1=predict(model,x_test_new);
    correlation1(o)=corr(pre_y_1,y_test);    
end
p=mean(correlation1>r_test);