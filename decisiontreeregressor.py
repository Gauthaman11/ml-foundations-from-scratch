import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


X=np.array([[500],[600],[700],[800],[900],[1000],[1100],[1200],[1300],[1400]])
y=np.array([20,25,30,35,40,45,50,55,60,65])

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeRegressor(criterion='squared_error',random_state=42,max_depth=3)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Actual Values: ",y_test)
print("Predicted Values: ",y_pred)