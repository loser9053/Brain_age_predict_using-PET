clear;
clc;
load('all.mat')
x_train=x_n(1:227,:);
y_train=y(1:227);
x_test=x_n(228:end,:);
y_test=y(228:end);
cvp = cvpartition(y_train,'k',10);
[fs,history] = sequentialfs(@class_SVR_poly,x_train,y_train,'cv',cvp,'direction','backward');

x_train=x_train(:,fs);
x_test=x_test(:,fs);

bestcv=0;
    for c=0.1:1:500
        for g=0.0001:0.0001:0.1
            cmd=['-s 3 -t 1 -d 2 -v 5 -c ', num2str(c),'-g ', num2str(g)];
            cv=svmtrain(y_train,x_train,cmd);
                if (cv>bestcv)
                    bestcv=cv;
                    bestc=c;
                    bestg=g;
                end
        end
    end
cmd=['-s 3 -t 1 -c ', num2str(bestc),'-g ',num2str(g)];
model=svmtrain(y_train,x_train,cmd);
[predicted_train, ~,~]=svmpredict(y_train,x_train,model);
[predicted_test, ~,~]=svmpredict(y_test,x_test,model);
mae_train=mean(abs(predicted_train-y_train));
mse_train=mean((predicted_train-y_train).^2);
nmse_train=mean((predicted_train-y_train).^2./y_train.^2);   
r_train=corr(predicted_train,y_train);

delta=predicted_train-y_train;
sigma=[ones(length(y_train),1),y_train];
B = regress(delta,sigma);
corrected_test_age=predicted_test-B(2)*y_test-B(1);

mae_test=mean(abs(corrected_test_age-y_test));
mse_test=mean((corrected_test_age-y_test).^2);
nmse_test=mean((corrected_test_age-y_test).^2./y_test.^2);   
r_test=corr(corrected_test_age,y_test);

perm=1000;
for o=1:perm
    pre_y_1=svmpredict(y_test,x_test(randperm(size(x_test,1)),:),model);
    correlation1(o)=corr(pre_y_1,y_test);    
end
p=mean(correlation1>r_test);