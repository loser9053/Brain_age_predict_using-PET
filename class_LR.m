function rmse = class_LR(x_train,y_train,x_test,y_test)
       X_train=[ones(size(x_train,1),1),x_train];
       [b,~,~,~,~]=regress(y_train,X_train);
       X_test=[ones(size(x_test,1),1),x_test];
       predicted_test=b'*X_test';
       rmse=sqrt(mean((predicted_test'-y_test).^2));