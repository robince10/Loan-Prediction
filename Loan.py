import pandas as pd
import numpy as np
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Gender'].fillna(train['Gender'].mode()[0], inplace = True)
train['Married'].fillna(train['Married'].mode()[0], inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace = True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace = True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace = True)

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isna().sum()
test.isna().sum()

from scipy import stats
stats.chi2_contingency(pd.crosstab(train['Gender'], train['Loan_Status']))
stats.chi2_contingency(pd.crosstab(train['Married'], train['Loan_Status']))
stats.chi2_contingency(pd.crosstab(train['Dependents'], train['Loan_Status']))
stats.chi2_contingency(pd.crosstab(train['Education'], train['Loan_Status']))
stats.chi2_contingency(pd.crosstab(train['Self_Employed'], train['Loan_Status']))
stats.chi2_contingency(pd.crosstab(train['Credit_History'], train['Loan_Status']))
stats.chi2_contingency(pd.crosstab(train['Property_Area'], train['Loan_Status']))

#Important Categorical Variables:- Married, Education, Credit History, Property_Area (Since, p value for each < 0.05)

train.drop(['Loan_ID', 'Gender', 'Dependents', 'Self_Employed'], axis = 1, inplace = True)
test.drop(['Loan_ID', 'Gender', 'Dependents', 'Self_Employed'], axis = 1, inplace = True)

train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']

sns.distplot(train['Total_Income'])

train['Total_Income'] = np.log(train['Total_Income'])
test['Total_Income'] = np.log(test['Total_Income'])

train.drop(['ApplicantIncome', 'CoapplicantIncome'], axis = 1, inplace = True)
test.drop(['ApplicantIncome', 'CoapplicantIncome'], axis = 1, inplace = True)

train['EMI'] = train['LoanAmount']/ train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount']/ test['Loan_Amount_Term']

train.drop(['LoanAmount', 'Loan_Amount_Term'], axis = 1, inplace = True)
test.drop(['LoanAmount', 'Loan_Amount_Term'], axis = 1, inplace = True)

train['Education'].replace('Graduate', 1, inplace = True)
test['Education'].replace('Graduate', 1, inplace = True)
train['Education'].replace('Not Graduate', 0, inplace = True)
test['Education'].replace('Not Graduate', 0, inplace = True)

train['Married'].replace('Yes', 1, inplace = True)
test['Married'].replace('Yes', 1, inplace = True)
train['Married'].replace('No', 0, inplace = True)
test['Married'].replace('No', 0, inplace = True)

train['Property_Area'].replace('Rural', 0, inplace = True)
test['Property_Area'].replace('Rural', 0, inplace = True)
train['Property_Area'].replace('Semiurban', 1, inplace = True)
test['Property_Area'].replace('Semiurban', 1, inplace = True)
train['Property_Area'].replace('Urban', 2, inplace = True)
test['Property_Area'].replace('Urban', 2, inplace = True)

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

X = train.drop('Loan_Status', axis = 1)
Y = train.Loan_Status

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
    sum += acc
avg_acc = sum/10

from sklearn.ensemble import AdaBoostClassifier

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
    sum += acc
avg_acc = sum/10

from sklearn.svm import SVC

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
    sum += acc
avg_acc = sum/10

paramgrid = [ {'n_estimators': [1,100,1],
'learning_rate':[0.01,0.1,0.02]},
]

from sklearn.model_selection import GridSearchCV

grid_search=GridSearchCV(AdaBoostClassifier(random_state=8),paramgrid)

from sklearn.model_selection import train_test_split 
x_train, x_cv, y_train, y_cv = train_test_split(X,Y, test_size =0.3, random_state=1)
grid_search.fit(x_train,y_train)

grid_search.best_estimator_

model = LogisticRegression(random_state = 8)
model.fit(X,Y)
Y_test = model.predict(test)

Y_test = pd.DataFrame(Y_test)

Y_test.to_csv('Submission.csv')