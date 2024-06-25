function rmse = class_SVR_rbf(xtrain,ytrain,xtest,ytest)
     model=svmtrain(ytrain,xtrain,'-s 3 -t 2 -c 1');
     [pre_y,~,~]=svmpredict(ytest,xtest,model);
     rmse=sqrt(mean((pre_y-ytest).^2));