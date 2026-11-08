#This is implementation of Support Vector Regression (SVR) model using scikit-learn library in Python.
#Libraries
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#dataset
X=np.array([[500],[600],[700],[800],[900],[1000],[1100],[1200],[1300],[1400]])
y=np.array([20,25,30,35,40,45,50,55,60,65])

#Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Scalling
X_scalling =StandardScaler()
y_scalling =StandardScaler()

#Scale X training data 
X_train_scaled=X_scalling.fit_transform(X_train)

#Scale y training data
y_train_scaled=y_scalling.fit_transform(y_train.reshape(-1,1)).ravel()

#Scale X testing data
X_test_scaled=X_scalling.transform(X_test)

#Create SVR model
model=SVR(kernel='rbf',C=1000,epsilon=0.1)

#Train the model
model.fit(X_train_scaled,y_train_scaled)

#Predictions
y_pred_scaled=model.predict(X_test_scaled)

#Inverse transform the predictions
y_pred=y_scalling.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

#Evalution

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R2 Score:",r2)


