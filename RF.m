clear;
clc;
load('all.mat')
x_train=x_n(1:227,:);
y_train=y(1:227);
x_test=x_n(228:end,:);
y_test=y(228:end);

indices=crossvalind('kfold',227,10);
tree_number=5:5:200;
rmse_CV=zeros(1,length(tree_number));
for i=1:length(tree_number)
    for j=1:10
        test=(indices==j);
        train=~test;
        tb = TreeBagger(tree_number(i), x_train(train,:), y_train(train,:),'Method','Regression');
        predicted_train_cv=predict(tb,x_train(test,:));
        rmse_CV(i)=rmse_CV(i)+sqrt(mean(predicted_train_cv-y_train(test)).^2);
    end
end
[~,index]=min(rmse_CV);
best_tree_num=tree_number(index);
save('best_tree_num.mat','best_tree_num');        

cvp = cvpartition(y_train,'k',10);
[fs,history] = sequentialfs(@class_RF,x_train,y_train,'cv',cvp,'direction','backward');

x_train=x_train(:,fs);
x_test=x_test(:,fs);


model=TreeBagger(best_tree_num,x_train,y_train,'Method','Regression');
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