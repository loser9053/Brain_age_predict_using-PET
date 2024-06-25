function rmse = class_RF(x_train,y_train,x_test,y_test)
       load best_tree_num.mat;
       model=TreeBagger(best_tree_num,x_train,y_train,'Method','Regression');
       predicted_test=predict(model,x_test);
       rmse=sqrt(mean((predicted_test-y_test).^2));