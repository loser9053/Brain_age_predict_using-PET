function rmse = class_SVR_poly(xtrain,ytrain,xtest,ytest)
     model=svmtrain(ytrain,xtrain,'-s 3 -t 1 -c 1 -d 2');
     [pre_y,~,~]=svmpredict(ytest,xtest,model);
     rmse=sqrt(mean((pre_y-ytest).^2));