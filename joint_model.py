import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
dataset= pd.read_csv("hourout_final.csv",header = 0)
print(dataset.info())

risk_difference = []

test_size_list=[0.2,0.3,0.4,0.5,0.6,0.65]
risk_difference1= []
for i in range(len(test_size_list)):
    val=float(test_size_list[i])
    X1= dataset.iloc[:, [6]] #Independent variable
    y = dataset.iloc[:, 17] #Dependent variable
    X2= dataset.iloc[:, [4]]
    X_train1, X_test1, y_train1, y_test = train_test_split(X1, y, test_size = val)
    X_train2, X_test2, y_train2, y_test = train_test_split(X2, y, test_size = val)
    classifier = LogisticRegression(random_state = 0,max_iter=400,solver = 'lbfgs')
#In the joint model after moltiplication we can simply add the log of probabities of two seperate x_test with very less covariates.
    classifier.fit(X_train1, y_train1)
    classifier.fit(X_train2,y_train2)
    x_prob1 = classifier.predict_log_proba(X_test1)
    x_prob2 = classifier.predict_log_proba(X_test2)
    X_prob = x_prob1.__add__(x_prob2)
    u = sum(X_prob)
    #print(u)
    v = u[0]
    w = u[1]
    z = v + w
    
    risk_diff = (v/z) -(w/z)
    risk_difference.append(risk_diff)
    
  
x_label3 = test_size_list
y_label3 = risk_difference
plt.plot(x_label3,y_label3,label ='Joint model' )
plt.scatter(x_label3,y_label3)
plt.show
    
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
dataset= pd.read_csv("hourout_final.csv",header = 0,na_values ='NaN')#dataset info

multicolinearity_check = dataset.corr()
test_size_list=[0.2,0.3,0.4,0.5,0.6,0.65]
risk_difference3= []
for i in range(len(test_size_list)):
    val=float(test_size_list[i])
    X = dataset.iloc[:, [ 3, 4, 5, 8 ,6, 7, 9]] #Independent variable
    y = dataset.iloc[:, 17] #Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = val)
    y_test = y_test.tolist()
    u = y_test.count(0)
    v = y_test.count(1)
    z = len(y)
    risk_diff = (v/(z)) -(u/(z))
    risk_difference3.append(risk_diff)
    
print('Risk_Difference for Sensitivity',risk_difference3)

x_label4 = test_size_list
y_label4 = risk_difference3
plt.plot(x_label4,y_label4,label = 'Sensitivity Analysis')
plt.scatter(x_label4,y_label4)
plt.legend()
plt.x_label4 = ('Size of the test sample')
plt.y_label4 = ('Risk Difference')
plt.show








