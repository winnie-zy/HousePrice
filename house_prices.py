#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

#Load the training data
df_train=pd.read_csv('train.csv')
#Load the test data
df_test=pd.read_csv('test.csv')

#Check the column headers
print(df_train.columns)
print(df_test.columns)

#Descriptive statistics summary
df_train['SalePrice'].describe()
#histogram
sns.distplot(df_train['SalePrice'])

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

#check if this variable is not relevant to the sale price.
#通过plot图看到有一定的线性关系，关联不是很强，可以放到后面feature selection再考虑
var = 'EnclosedPorch'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#check duplicated rows
df_train[df_train.duplicated()==True]

#check column data types
res = df_train.dtypes
print(res[res == np.dtype('int64')])
print(res[res == np.dtype('bool')])
print(res[res == np.dtype('object')])
print(res[res == np.dtype('float64')])

#standardize
print(df_train["LotConfig"].unique())

# feature scaling, only apply to numeric data.fit_transform()必须是二维数组，可使用注释部分
sc_X = StandardScaler()
X_train = sc_X.fit_transform(df_train[["GrLivArea","SalePrice"]])
sns.distplot(X_train[:,1],fit=norm)
# X_train = sc_X.fit_transform(df_train[["SalePrice"]])
# sns.distplot(X_train[:,0],fit=norm)

#histogram and normal probability plot,transformation前的'GrLivArea'展示
sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

#data transformation,including train and test dataset
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])#Log computation
sns.distplot(df_train['GrLivArea'], fit=norm)#display result
res = stats.probplot(df_train['GrLivArea'], plot=plt)

df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
sns.distplot(df_test['GrLivArea'], fit=norm)

#missing data df_train.isnull().sum()对为NULL的计数
total = df_train.isnull().sum().sort_values(ascending=False)
total_test=df_test.isnull().sum().sort_values(ascending=False)
#df.count() tells us the number of non-NaN values based on all coloumn names.
#df['coloumn name'].count() gives the the number of non-NaN values for the specific coloumn
#df_train.isnull().count() equals len(df_train)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
percent_test=(df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)#把缺失率超过15%的列删除掉，index为drop的列名，1为以列为准
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)#drop某一行，该行‘Electrical’为NULL
df_test = df_test.drop((missing_data_test[missing_data_test['Percent'] > 0.15]).index,1)#把缺失率超过15%的列删除掉，index为drop的列名，1为以列为准
#df_test中去除某字段为空的那行记录
df_train['BsmtFinSF2'].corr(df_train['BsmtFinSF1'])
df_train['BsmtUnfSF'].corr(df_train['BsmtFinSF1'])
df_train['GarageArea'].corr(df_train['GarageCars'])
df_train['BsmtHalfBath'].corr(df_train['BsmtFullBath'])
df_test = df_test.drop(df_test.loc[df_test['BsmtFinSF2'].isnull()].index)
# df_test = df_test.drop(df_test.loc[df_test['BsmtFinSF1'].isnull()].index)
df_test = df_test.drop(df_test.loc[df_test['Exterior2nd'].isnull()].index)
# df_test = df_test.drop(df_test.loc[df_test['BsmtUnfSF'].isnull()].index)
# df_test = df_test.drop(df_test.loc[df_test['TotalBsmtSF'].isnull()].index)
df_test = df_test.drop(df_test.loc[df_test['SaleType'].isnull()].index)
# df_test = df_test.drop(df_test.loc[df_test['Exterior1st'].isnull()].index)
df_test = df_test.drop(df_test.loc[df_test['KitchenQual'].isnull()].index)
df_test = df_test.drop(df_test.loc[df_test['GarageArea'].isnull()].index)
# df_test = df_test.drop(df_test.loc[df_test['GarageCars'].isnull()].index)
df_test = df_test.drop(df_test.loc[df_test['BsmtHalfBath'].isnull()].index)
# df_test = df_test.drop(df_test.loc[df_test['BsmtFullBath'].isnull()].index)
#if not drop, we can impute,合并已有数值，填补missing value
# check correlation with LotArea,if not drop, we can impute new value for missing

df_train['LotFrontage'].corr(df_train['LotArea'])

df_train['SqrtLotArea']=np.sqrt(df_train['LotArea'])
df_train['LotFrontage'].corr(df_train['SqrtLotArea'])#计算两列的相关性，>0.5就是highly coorelated

cond = df_train['LotFrontage'].isnull()
df_train["LotFrontage"][cond]=df_train["SqrtLotArea"][cond]
print(df_train["LotFrontage"].isnull().sum())

#flag the missing data as missing,this is for string type
mis=df_train['GarageType'].isnull()
df_train["GarageType"][mis]="Missing"
df_train["GarageType"].unique()

#identify the outliers
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 4))
axes = np.ravel(axes)
col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']
for i, c in zip(range(5), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='r')
plt.show()
#delete outliers
print(df_train.shape)
df_train = df_train[df_train['GrLivArea'] < 4500]
df_train = df_train[df_train['LotArea'] < 100000]
df_train = df_train[df_train['TotalBsmtSF'] < 3000]
df_train = df_train[df_train['1stFlrSF'] < 2500]
df_train = df_train[df_train['BsmtFinSF1'] < 2000]

print(df_train.shape)

for i, c in zip(range(5,10), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='b')
plt.show()

#Now feature selection part
print(df_train.info()) #still 76 features remain

#check distribution of all the inputs
df_train.hist(figsize=(20, 20), bins=20)
plt.show()

#Based on the distribution results, check suspicious inputs
#3SsnPorch has too many zeros
df_train['3SsnPorch'].describe()

#BedroomAbvGr is not normal distributed
np.unique(df_train['BedroomAbvGr'].values)##df_train['BedroomAbvGr'].unique()
df_train.groupby('BedroomAbvGr').count()['Id']#对每一个value count总数

#Two Basement bathroom variables, can be merged together FullBath
df_train.corr()['BsmtFullBath']['BsmtHalfBath']
df_train['BsmtFullBath'].corr(df_train['BsmtHalfBath'])
df_train.groupby('BsmtFullBath').count()['Id']
df_train.groupby('BsmtHalfBath').count()['Id']
df_train.groupby('FullBath').count()['Id']
df_train.groupby('HalfBath').count()['Id']

#完全相同概念的变量可以直接相加，相乘处理
df_train['BsmtBathroom']=df_train['BsmtFullBath']+ df_train['BsmtHalfBath'] 
df_train['Bathroom']=df_train['FullBath']+ df_train['HalfBath'] 

#Four basement area variables. Three can be dropped
df_train[['TotalBsmtSF', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF']].head()

# Three more porch related variables. We can merge them togher or just keep one.
df_train[['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']].describe()
df_train['Porch']=df_train['EnclosedPorch']+df_train['ScreenPorch']+df_train['3SsnPorch']
#Garage area and cars must be correlated. Use area or cars?
df_train.corr()['GarageArea']['GarageCars']

#garage year built can be also dropped because we have a varaible: house year built.
df_train.corr()['GarageYrBlt']['YearBuilt']

#KitchenAbvGr can be dropped as there are too many 1s，基本都是一个厨房，对sales-price影响不大，可以去掉。
df_train['KitchenAbvGr'].describe() 

#Lot area has a small proportion houses which have large area, need to be filtered.查看LotArea是否满足正太分布
sns.distplot(df_train['LotArea'], bins=100) #can filter Lot Area above 50000 to 50001

#Now let's check the correlation matrix，'SalePrice'对每一个input的关系
#可以找到那些变量与SalePrice的相关性，分析时要去除关联性很高的变量重复出现，防止变量的多重贡献性
df_train.corr()['SalePrice'].sort_values()

#YearBuilt and YearRemodAdd seems correlated
df_train.corr()['YearBuilt']['YearRemodAdd'] 

#Only select numeric variables (including SalePrice)
num_attrs = df_train.select_dtypes([np.int64, np.float64]).columns.values
df_train_num= df_train[num_attrs]

#Merge two bathroom variables,如上
# df_train_num['Bath']= df_train_num['BsmtFullBath'] + df_train_num['BsmtHalfBath'] 

#Remove the above variables
df_train_num=df_train_num.drop(['Id','3SsnPorch','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','EnclosedPorch','ScreenPorch','GarageCars',
                                'GarageYrBlt','KitchenAbvGr','YearRemodAdd','BsmtFullBath', 'BsmtHalfBath','FullBath','HalfBath'],axis=1)
#Get the correlation matrix
corr_1 = df_train_num.corr()
corr_1 = corr_1.applymap(lambda x : 1 if x > 0.7 else -1 if x < -0.7 else 0)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_1, vmax=1, center=0,vmin=-1 ,  square=True, linewidths=.005)
plt.show()
#通过观察热点图，发现TotalBsmtSF，1stFlrSF相关性很大，需要去掉其中一个，避免多重贡献性
df_train_num.corr()['TotalBsmtSF']['1stFlrSF']
df_train_num.corr()['TotRmsAbvGrd']['GrLivArea']
df_train_num.corr()['Bathroom']['GrLivArea']

#Identify two correlated variables,同样去掉2ndFlrSF
df_train_num=df_train_num.drop(['TotRmsAbvGrd','Bathroom'],axis=1)
df_train_num=df_train_num.drop(['1stFlrSF'],axis=1)
#list the correlation values
df_train_num.corr()['SalePrice'].sort_values()

#select correlation >0.3（变量与output关联）
df_train_num=df_train_num[df_train_num.columns[df_train_num.corr()['SalePrice']>0.3]]
df_train_num.columns  

#Build the model
from sklearn import linear_model
reg = linear_model.LinearRegression()

#Split the input and output
df_train_num_x=df_train_num.drop('SalePrice',axis=1) 
df_train_num_y=df_train_num['SalePrice']
#填充为NULL的变量，使用该列均值代替。一对样本中增加一个均值，对样本变化很小
missing=df_train['MasVnrArea'].isnull()
df_train_num_x['MasVnrArea'][missing]=np.mean(df_train_num_x['MasVnrArea'])
#Train the model
reg.fit(df_train_num_x, df_train_num_y)#把input.output放到模型中训练

#Check the model coefficients
print('Coefficients: \n', reg.coef_)

#Get the prediction based on the training dataset
preds = reg.predict(df_train_num_x)

#Check the training dataset prediction performance
from sklearn import metrics
#Mean Absolute Error 
print('MAE:', metrics.mean_absolute_error(df_train_num_y, preds))
#Mean Squared Error 平方
print('MSE:', metrics.mean_squared_error(df_train_num_y, preds))
#Root Mean Squared Error 开方
print('RMSE:', np.sqrt(metrics.mean_squared_error(df_train_num_y, preds)))

#Plot the predictions and actuals。预测的salesprice和实际数值进行比较
plt.scatter(df_train_num_y,preds)

#Check the error.用实际价格-预测价格=error.linear regression模型的error分布满足正态分布 error=df_train_num_y-preds
sns.distplot((df_train_num_y-preds),bins=35)

#Load the test data
df_test=pd.read_csv('test.csv')
# df_test['Bath']= df_test['BsmtFullBath'] + df_test['BsmtHalfBath'] 
df_test['BsmtBathroom']=df_test['BsmtFullBath']+ df_test['BsmtHalfBath'] 
df_test['Bathroom']=df_test['FullBath']+ df_test['HalfBath']
df_test['Porch']=df_test['EnclosedPorch']+df_test['ScreenPorch']+df_test['3SsnPorch']
df_test_num= df_test[['LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF',
       '2ndFlrSF', 'GrLivArea', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF','Id']]

#IMPORTANT: All the feature engineering & data cleaning steps we have done to the training variables, we have to do the same for the test dataset!!
#Before we can feed the data into our model, we have to check missing values again. Otherwise the code will give you an error.
df_test_num.isnull().sum()
df_test_num['MasVnrArea']=df_test_num['MasVnrArea'].fillna(np.mean(df_test_num['MasVnrArea']))##用mean数值进行填充
# df_test_num['GarageArea']=df_test_num['GarageArea'].fillna(np.mean(df_test_num['GarageArea']))
# df_test_num_x['MasVnrArea'][missing]=np.mean(df_test_num_x['MasVnrArea'])

#Predict the results for test dataset
submit= pd.DataFrame()
submit['Id'] = df_test_num['Id']
#select features 
# preds_out = reg.predict(df_test_num[['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea','FullBath', 'GarageArea']])
preds_out=reg.predict(df_test_num[['LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF',
       '2ndFlrSF', 'GrLivArea', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']])
submit['SalePrice'] = preds_out
#final submission  pandas方法，将dataframe直接存
submit.to_csv('test_submit.csv', index=False)

#Check output
#check yearly alignment
df_train['preds']=preds
df_yearly=df_train[['SalePrice','preds','YearBuilt']].groupby('YearBuilt').mean()
sns.lineplot(data=df_yearly)

#check Rates the overall material and finish of the house
df_yearly1=df_train[['SalePrice','preds','OverallQual']].groupby('OverallQual').mean()
sns.lineplot(data=df_yearly1)

#check Rates the overall condition of the house
df_yearly2=df_train[['SalePrice','preds','OverallCond']].groupby('OverallCond').mean()
sns.lineplot(data=df_yearly2)

#check Bedrooms
df_yearly3=df_train[['SalePrice','preds','BedroomAbvGr']].groupby('BedroomAbvGr').mean()
sns.lineplot(data=df_yearly3)
