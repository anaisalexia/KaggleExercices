import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#load data
data_set= pd.read_csv('C:/pythonProjectKaggle/space_ship/train.csv')

#analyse/visualise data
# print(data_set.shape)
# print(data_set.columns)
print(data_set.info())

#number of unique values
number={}
data_numerical=[]
data_drop=[]
data_keep=[]
for i in data_set.columns:
    if data_set[i].nunique( dropna=True) > 5 and data_set[i].dtype == 'object':
        data_drop.append(i)
    elif data_set[i].nunique( dropna=True) <= 5  and data_set[i].dtype in ['object']:
        data_keep.append(i)

    elif data_set[i].dtype in ['int64','float64']:
        data_numerical.append(i)

print( "numerical columns : {} \n columns to keep : {} \n columns to drop : {}".format(data_numerical,data_keep,data_drop))

data_set['Expense']= data_set[[i for i in data_numerical if i != 'Age' ]].sum(axis=1)

# print(data_set[data_numerical].head())
# print(data_set[['Age']+data_keep].head(10))
# print(data_numerical)

#Get read of Na numerical values
imput = SimpleImputer()
data_set[data_numerical]=imput.fit_transform(data_set[data_numerical])

data_numerical_keep=['Age','Expense']

#Encoding and getting rid of NA for non numerical values
imput2 = SimpleImputer(strategy="most_frequent")
data_set[data_keep]=imput2.fit_transform(data_set[data_keep])

enc=LabelEncoder()
for i in data_keep:
    data_set[i]=enc.fit_transform(data_set[i])

# print(data_set[data_keep+data_numerical_keep].head(10))

#split the data

y=data_set['Transported']
X=data_set[data_keep+data_numerical_keep].copy(deep=True)
# X.drop('Transported',axis=1,inplace=True)

# #Spliting the data to evaluate the model prior to submission
# X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.9, test_size=0.1)
#
# #Model
#
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# model.fit(X_train,y_train)
#
# #Prediction and evaluation
# pred_valid=model.predict(X_valid)
# score= mean_absolute_error(y_valid,pred_valid)
#
# print("score MAE: {}".format(score))

#For submission
#Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X,y)

#preparing test data
data_test= pd.read_csv('C:/pythonProjectKaggle/space_ship/test.csv')

data_test['Expense']= data_test[[i for i in data_numerical if i != 'Age' ]].sum(axis=1)
data_test[data_numerical]=imput.transform(data_test[data_numerical])
data_numerical_keep=['Age','Expense']

data_keep_test= data_keep

data_test[data_keep_test]=imput2.fit_transform(data_test[data_keep_test])
for i in data_keep_test:
    data_test[i]=enc.fit_transform(data_test[i])

X_test=data_test[data_keep_test+data_numerical_keep].copy(deep=True)

#Prediction
pred_valid=model.predict(X_test)
# print("prediction",pred_valid)
preds=pd.DataFrame(pred_valid,columns=['Transported'])
preds=preds.join(data_test['PassengerId'])
# print(preds.head())
preds.to_csv(r"./pred_space.csv",index=False)

