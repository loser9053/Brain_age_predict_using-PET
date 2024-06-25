function rmse = class_ridge(x_train,y_train,x_test,y_test)
       load best_ridge_num.mat;
       b=ridge(y_train,x_train,ridge_num);
       predicted_test=b'*x_test';
       rmse=sqrt(mean((predicted_test'-y_test).^2));