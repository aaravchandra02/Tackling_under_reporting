import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.metrics import confusion_matrix
from pylab import scatter, show, legend, xlabel, ylabel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

dataset= pd.read_csv("hourout_final.csv",header = 0,na_values ='NaN')
#print(dataset.info())
multicolinearity_check = dataset.corr()

#In both underreporting and spliting up of size, the size of the sample is varied and affected.
#So under reproting is similar to Splitting of data


#logistic Regression model

test_size_list=[0.2,0.3,0.4,0.5,0.6,0.65]
risk_difference1= []
for i in range(len(test_size_list)):
    val=float(test_size_list[i])
    X = dataset.iloc[:, [4,6,8]] #Independent variable
    y = dataset.iloc[:, 17] #Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = val)
   
    #logistic regression
    classifier = LogisticRegression(random_state = 0,max_iter=400,solver = 'lbfgs')
    classifier.fit(X_train, y_train)
    x_prob = classifier.predict_proba(X_test)
    #print('probality ',x_prob)
    y_pred = classifier.predict(X_test)
    #print('Prediction', y_pred)
    
    #K_fold validation & model accuracy
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    model_accuracy = accuracies.mean()
    model_standard_deviation = accuracies.std()
    #print('model Accuracy', model_accuracy)
    
    #mutual score
    mut_score = mutual_info_score(dataset.iloc[:, 3],y)
    print("mutual Score for A and Covariates for Logistic   ", mut_score)
    
    #confusion matrix
    cf_m = confusion_matrix(y_test, y_pred)
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    riskdiff = (cf_m[0][0]/(cf_m[0][0]+cf_m[0][1]))-(cf_m[1][0]/(cf_m[1][0]+cf_m[1][1]))
    risk_difference1.append(riskdiff)



print('Risk Difference1 for Logistic',risk_difference1)


#graph between testsize and riskdifference 
x_label1 = test_size_list
y_label1 = risk_difference1
plt.plot(x_label1,y_label1,label = 'Logistic')
plt.legend()
plt.xlabel = ('Size of the test sample')
plt.ylabel = ('Risk Difference')
plt.show

#Propensity Model

test_size_list=[0.2,0.3,0.4,0.5,0.6,0.65]
risk_difference2= []
for i in range(len(test_size_list)):
    val=float(test_size_list[i])
    X = dataset.iloc[:, [4, 6]]#Independent variable
    y = dataset.iloc[:, 8]#Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = val)
    
    #logistic regression
    classifier = LogisticRegression(random_state = 0,max_iter=400,solver = 'lbfgs')# lbfgs condition
    classifier.fit(X_train, y_train)
    x_prob = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    #print('Prediction', y_pred)
    
    #K_fold validation & model accuracy
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    model_accuracy = accuracies.mean()
    model_standard_deviation = accuracies.std()
    #print('model Accuracy', model_accuracy)
    
    #mutual score
    mut_score = mutual_info_score(dataset.iloc[:, 6],y)
    print("mutual Score for A and Covariates for propensity ", mut_score)
    
    #confusion matrix
    from sklearn.metrics import confusion_matrix,mutual_info_score
    cf_m = confusion_matrix(y_test, y_pred)
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    riskdiff = (cf_m[0][0]/(cf_m[0][0]+cf_m[0][1]))-(cf_m[1][0]/(cf_m[1][0]+cf_m[1][1]))
    risk_difference2.append(riskdiff)



print('Risk Differnce for propensity ',risk_difference2)

#graph between testsize and riskdifference 
x_label2 = test_size_list
y_label2 = risk_difference2
plt.plot(x_label2,y_label2,label = 'Propensity')
plt.legend()
plt.xlabel = ('Size of the test sample')
plt.ylabel = ('Risk Difference')
plt.show



#Sensitivity analysis

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
dataset= pd.read_csv("hourout_final.csv",header = 0,na_values ='NaN')#dataset info

multicolinearity_check = dataset.corr()
test_size_list=[0.2,0.3,0.4,0.5,0.6,0.65]
risk_difference3= []
for i in range(len(test_size_list)):
    val=float(test_size_list[i])
    X = dataset.iloc[:, [3, 4, 5, 8 ,6, 7, 9]] #Independent variable
    y = dataset.iloc[:, 17] #Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = val)
    y_test = y_test.tolist()
    u = y_test.count(0)
    v = y_test.count(1)
    z = len(y)
    risk_diff = (v/(z)) -(u/(z))
    risk_difference3.append(risk_diff)
    
print('Risk_Difference for Sensitivity',risk_difference3)

x_label3 = test_size_list
y_label3 = risk_difference3
plt.plot(x_label3,y_label3,label = 'Sensitivity analysis')
plt.legend()
plt.xlabel = ('Size of the test sample')
plt.ylabel = ('Risk Difference')
plt.show

#joint_distribution

    
    










