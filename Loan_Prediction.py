
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('C:/Users/praj/Downloads/Loan_Prediction/train_ctrUa4K.csv')
test = pd.read_csv('C:/Users/praj/Downloads/Loan_Prediction/test_lAUu6dG.csv')


# In[3]:


train.head()


# In[4]:


train.isnull().sum()


# In[5]:


train['Gender'].value_counts()


# In[39]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace = True)


# In[40]:


train['Married'].fillna(train['Married'].mode()[0], inplace = True)


# In[41]:


train['Dependents'].value_counts()


# In[42]:


train['Dependents'].fillna(train['Dependents'].mode()[0], inplace = True)


# In[43]:


train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace = True)


# In[44]:


train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace = True)


# In[45]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace = True)


# In[46]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace = True)


# In[47]:


train.isnull().sum()


# In[48]:


test.isnull().sum()


# In[49]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[50]:


test.isnull().sum()


# In[51]:


#Categorical variables are gender, married, dependents, education, self_employed, credit_history, property_area
from scipy import stats


# In[52]:


stats.chi2_contingency(pd.crosstab(train['Gender'], train['Loan_Status']))


# In[53]:


stats.chi2_contingency(pd.crosstab(train['Married'], train['Loan_Status']))


# In[54]:


stats.chi2_contingency(pd.crosstab(train['Dependents'], train['Loan_Status']))


# In[55]:


stats.chi2_contingency(pd.crosstab(train['Education'], train['Loan_Status']))


# In[56]:


stats.chi2_contingency(pd.crosstab(train['Self_Employed'], train['Loan_Status']))


# In[57]:


stats.chi2_contingency(pd.crosstab(train['Credit_History'], train['Loan_Status']))


# In[58]:


stats.chi2_contingency(pd.crosstab(train['Property_Area'], train['Loan_Status']))


# In[59]:


#Important Categorical Variables:- Married, Education, Credit History, Property_Area (Since, p value for each < 0.05)


# In[60]:


train.drop(['Loan_ID', 'Gender', 'Dependents', 'Self_Employed'], axis = 1, inplace = True)
test.drop(['Loan_ID', 'Gender', 'Dependents', 'Self_Employed'], axis = 1, inplace = True)


# In[61]:


train.shape


# In[62]:


test.shape


# In[63]:


train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']


# In[64]:


test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']


# In[65]:


sns.distplot(train['Total_Income'])


# In[66]:


train['Total_Income'] = np.log(train['Total_Income'])
test['Total_Income'] = np.log(test['Total_Income'])


# In[67]:


sns.distplot(train['Total_Income'])


# In[68]:


train.drop(['ApplicantIncome', 'CoapplicantIncome'], axis = 1, inplace = True)


# In[69]:


test.drop(['ApplicantIncome', 'CoapplicantIncome'], axis = 1, inplace = True)


# In[75]:


train.shape


# In[76]:


test.shape


# In[72]:


train['EMI'] = train['LoanAmount']/ train['Loan_Amount_Term']


# In[73]:


test['EMI'] = test['LoanAmount']/ test['Loan_Amount_Term']


# In[74]:


train.drop(['LoanAmount', 'Loan_Amount_Term'], axis = 1, inplace = True)
test.drop(['LoanAmount', 'Loan_Amount_Term'], axis = 1, inplace = True)


# In[77]:


train.head()


# In[78]:


train.Education.value_counts()


# In[79]:


train['Education'].replace('Graduate', 1, inplace = True)
test['Education'].replace('Graduate', 1, inplace = True)
train['Education'].replace('Not Graduate', 0, inplace = True)
test['Education'].replace('Not Graduate', 0, inplace = True)


# In[80]:


train['Married'].replace('Yes', 1, inplace = True)
test['Married'].replace('Yes', 1, inplace = True)
train['Married'].replace('No', 0, inplace = True)
test['Married'].replace('No', 0, inplace = True)


# In[81]:


train['Property_Area'].value_counts()


# In[82]:


train['Property_Area'].replace('Rural', 0, inplace = True)
test['Property_Area'].replace('Rural', 0, inplace = True)
train['Property_Area'].replace('Semiurban', 1, inplace = True)
test['Property_Area'].replace('Semiurban', 1, inplace = True)
train['Property_Area'].replace('Urban', 2, inplace = True)
test['Property_Area'].replace('Urban', 2, inplace = True)


# In[83]:


train.head()


# In[84]:


test.head()


# In[85]:


train.EMI.max()


# In[86]:


train.EMI.min()


# In[87]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


# In[88]:


X = train.drop('Loan_Status', axis = 1)
Y = train.Loan_Status


# In[89]:


X[:5]


# In[90]:


Y[:5]


# In[95]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = LogisticRegression(random_state = 8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10


# In[96]:


avg_acc


# In[97]:


pred_test = model.predict(test) 


# In[99]:


results = pd.read_csv('C:/Users/praj/Downloads/Loan_Prediction/sample_submission_49d68Cx.csv')


# In[100]:


results['Loan_Status'] = pred_test


# In[101]:


results.head()


# In[102]:


results.tail()


# In[104]:


results['Loan_Status'].value_counts()


# In[105]:


pwd


# In[106]:


results.to_csv('results.csv', index = False)


# In[107]:


from sklearn.ensemble import AdaBoostClassifier


# In[108]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = AdaBoostClassifier(n_estimators = 100, random_state = 8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10


# In[109]:


avg_acc


# In[110]:


pred_test = model.predict(test)


# In[111]:


results['Loan_Status'] = pred_test


# In[112]:


results.to_csv('AdaBoost.csv', index = False)


# In[113]:


from sklearn.svm import SVC


# In[114]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = SVC(random_state = 8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10


# In[115]:


avg_acc


# In[116]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('SVC.csv', index = False)


# In[117]:


from sklearn.tree import DecisionTreeClassifier


# In[120]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = DecisionTreeClassifier(criterion='gini', random_state = 8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10


# In[121]:


avg_acc


# In[125]:


from sklearn.neighbors.nearest_centroid import NearestCentroid


# In[127]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = NearestCentroid()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[128]:


from sklearn.neighbors import KNeighborsClassifier


# In[130]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = KNeighborsClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[131]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('KNN.csv', index = False)


# In[132]:


from sklearn.naive_bayes import GaussianNB


# In[133]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[134]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('Naive_Bayes.csv', index = False)


# In[136]:


from sklearn.linear_model import SGDClassifier


# In[137]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = SGDClassifier(loss = 'modified_huber', random_state = 8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[138]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('SGD.csv', index = False)


# In[139]:


from sklearn.ensemble import RandomForestClassifier


# In[140]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = RandomForestClassifier(n_estimators = 70, random_state = 8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[141]:


from sklearn.model_selection import GridSearchCV


# In[142]:


paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}


# In[143]:


grid_search=GridSearchCV(RandomForestClassifier(random_state=8),paramgrid)


# In[144]:


from sklearn.model_selection import train_test_split 
x_train, x_cv, y_train, y_cv = train_test_split(X,Y, test_size =0.3, random_state=1)
grid_search.fit(x_train,y_train)


# In[145]:


grid_search.best_estimator_


# In[146]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=3, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=21, n_jobs=None,
            oob_score=False, random_state=8, verbose=0, warm_start=False)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[147]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('RF.csv', index = False)


# In[148]:


grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}


# In[150]:


logreg = LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)


# In[151]:


from sklearn.model_selection import train_test_split 
x_train, x_cv, y_train, y_cv = train_test_split(X,Y, test_size =0.3, random_state=1)
logreg_cv.fit(x_train,y_train)


# In[152]:


logreg_cv.best_estimator_


# In[153]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[154]:


avg_acc


# In[155]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('logreg.csv', index = False)


# In[179]:


paramgrid = [ {'n_estimators': [1,100,1],
'learning_rate':[0.01,0.1,1]},
]


# In[180]:


grid_search=GridSearchCV(AdaBoostClassifier(random_state=8),paramgrid)


# In[181]:


from sklearn.model_selection import train_test_split 
x_train, x_cv, y_train, y_cv = train_test_split(X,Y, test_size =0.3, random_state=1)
grid_search.fit(x_train,y_train)


# In[182]:


grid_search.best_estimator_


# In[183]:


i = 1
sum = 0
kf = StratifiedKFold(n_splits=10, random_state = 8, shuffle=True)
for i, j in kf.split(X, Y):
    X_train = X.loc[i]
    X_cv = X.loc[j]
    Y_train = Y.loc[i]
    Y_cv = Y.loc[j]
    model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.01, n_estimators=1, random_state=8)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_cv)
    acc = accuracy_score(Y_cv, Y_pred)
    print('The accuracy for this strata is: {}'.format(acc))
    i = i + 1
    sum += acc
avg_acc = sum/10
print(avg_acc)


# In[184]:


pred_test = model.predict(test)
results['Loan_Status'] = pred_test
results.to_csv('AdaBoost2.csv', index = False)

