function rmse = class_RT(x_train,y_train,x_test,y_test)
       model=RegressionTree.fit(x_train,y_train);
       predicted_test=predict(model,x_test);
       rmse=sqrt(mean((predicted_test-y_test).^2));