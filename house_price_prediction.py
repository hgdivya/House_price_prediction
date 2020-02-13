import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 
import warnings
from scipy.stats import skew
from scipy import stats 
from scipy.stats import norm 
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook' )
sns.set_palette("BrBG", 7)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

trainset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")

# The numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(trainset.shape))
print("The test data size before dropping Id feature is : {} ".format(testset.shape))

train_ID = trainset['Id']
test_ID = testset['Id']

# dropping of the 'Id' column since it's unnecessary for the prediction process.
trainset.drop("Id", axis = 1, inplace = True)
testset.drop("Id", axis = 1, inplace = True)

#After dropping id
print("\nThe train data size after dropping Id feature is : {} ".format(trainset.shape)) 
print("The test data size after dropping Id feature is : {} ".format(testset.shape))

trainset.head()

testset.head()

# Getting Description of target variable
trainset['SalePrice'].describe()

# Plot normal distribution plot
sns.set_palette("BrBG", 7)
sns.distplot(trainset['SalePrice'] , fit=norm);

# to get the fitted parameters used by the function
(mu, sigma) = norm.fit(trainset['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#probability plot to check the fitting of saleprice over normal distribution 
fig = plt.figure()
res = stats.probplot(trainset['SalePrice'], plot=plt)
plt.show()

#check for skewness,kurtosis and print it
print("Skewness: ", trainset['SalePrice'].skew())
print("Kurtosis: ", trainset['SalePrice'].kurt())

#Categorical Data
trainset.select_dtypes(include=['object']).columns

# Numerical Data
trainset.select_dtypes(include=['int64','float64']).columns

cat_len = len(trainset.select_dtypes(include=['object']).columns)
num_len = len(trainset.select_dtypes(include=['int64','float64']).columns)
print('Overall Features: ', cat_len, 'categorical', '+',
      num_len, 'numerical', '=', cat_len+num_len, 'features')

# To create correlation Matrix Heatmap
cor_mat = trainset.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cor_mat, vmax=.8, square=True);

# Top 10 Heatmap with correlation coeff values
k = 10 
colms = cor_mat.nlargest(k, 'SalePrice')['SalePrice'].index
coef = np.corrcoef(trainset[colms].values.T)
sns.set(font_scale=1.25)
htmp = sns.heatmap(coef, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=colms.values, xticklabels=colms.values)
plt.show()
#10 most correlated features
mostcor = pd.DataFrame(colms)
mostcor.columns = ['Most Correlated Features']
mostcor

# comparing the most 10 correlated feature with sale price to remove outliners done using the plots
# Overall quality vs sale price (since unique values are less box plot used)
f, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x=trainset['OverallQual'], y=trainset['SalePrice'], palette="pastel")

# Living Area vs Sale Price
#since unique values are more in this feature we use joint scatter plot to determine outliers 
sns.jointplot(x=trainset['GrLivArea'], y=trainset['SalePrice'], kind='scatter', color="g")

# Removing outliers manually (Two points in the bottom right which are leading to tail)
trainset = trainset.drop(trainset[(trainset['GrLivArea']>4000) 
                         & (trainset['SalePrice']<300000)].index).reset_index(drop=True)

# Living Area vs Sale Price
sns.jointplot(x=trainset['GrLivArea'], y=trainset['SalePrice'], kind='scatter',color="g")

# Garage Area vs Sale Price
sns.boxplot(x=trainset['GarageCars'], y=trainset['SalePrice'], palette="pastel" )

# Removing outliers manually (More than 4-cars, less than $300k)
trainset = trainset.drop(trainset[(trainset['GarageCars']>3) 
                         & (trainset['SalePrice']<300000)].index).reset_index(drop=True)

# after removing outliners Garage Area vs Sale Price
sns.boxplot(x=trainset['GarageCars'], y=trainset['SalePrice'], palette="pastel" )

# Garage Area vs Sale Price
sns.jointplot(x=trainset['GarageArea'], y=trainset['SalePrice'], kind='scatter', color="g")

# Removing outliers manually (More than 1000 sqft, less than $300k)
trainset = trainset.drop(trainset[(trainset['GarageArea']>1000) 
                         & (trainset['SalePrice']<300000)].index).reset_index(drop=True)

# Garage Area vs Sale Price
sns.jointplot(x=trainset['GarageArea'], y=trainset['SalePrice'], kind='scatter', color="g")

# Basement Area vs Sale Price
sns.jointplot(x=trainset['TotalBsmtSF'], y=trainset['SalePrice'], kind='scatter', color="g")

# First Floor Area vs Sale Price
sns.jointplot(x=trainset['1stFlrSF'], y=trainset['SalePrice'], kind='scatter',color="g")

# Total Rooms vs Sale Price
sns.boxplot(x=trainset['TotRmsAbvGrd'], y=trainset['SalePrice'], palette="pastel" )

#Year built vs Sale Price
f, ax = plt.subplots(figsize=(18, 9))
sns.boxplot(x=trainset['YearBuilt'], y=trainset['SalePrice'], palette="pastel" )
plt.xticks(rotation=90);

# Combining Datasets
n_train = trainset.shape[0]
n_test = testset.shape[0]
y_train = trainset.SalePrice.values
alldata = pd.concat((trainset, testset)).reset_index(drop=True)
alldata.drop(['SalePrice'], axis=1, inplace=True)
print("Trainset data size is : {}".format(trainset.shape))
print("Testset data size is : {}".format(testset.shape))
print("Combined dataset size is : {}".format(alldata.shape))

# Find Missing Ratio of Dataset
alldata_na = (alldata.isnull().sum() / len(alldata)) * 100
alldata_na = alldata_na.drop(alldata_na[alldata_na == 0].index).sort_values(ascending=False)[:30] # drop which has 0 percent missing
misdata = pd.DataFrame({'Missing Ratio' :alldata_na})
misdata

# Percent missing data by feature plotted
f, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x=alldata_na, y=alldata_na.index, palette="pastel")
plt.xlabel('Percentage of missing values', fontsize=15)
plt.ylabel('Features with missing values', fontsize=15)
plt.title('Percentage of missing data by feature', fontsize=15)

alldata["PoolQC"] = alldata["PoolQC"].fillna("None")
alldata["MiscFeature"] = alldata["MiscFeature"].fillna("None")
alldata["Alley"] = alldata["Alley"].fillna("None")
alldata["Fence"] = alldata["Fence"].fillna("None")
alldata["FireplaceQu"] = alldata["FireplaceQu"].fillna("None")
alldata["LotFrontage"] = alldata.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for cl in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    alldata[cl] = alldata[cl].fillna('None')
for cl in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    alldata[cl] = alldata[cl].fillna(0)
for cl in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    alldata[cl] = alldata[cl].fillna(0)
for cl in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    alldata[cl] = alldata[cl].fillna('None')
alldata["MasVnrType"] = alldata["MasVnrType"].fillna("None")
alldata["MasVnrArea"] = alldata["MasVnrArea"].fillna(0)
alldata['MSZoning'] = alldata['MSZoning'].fillna(alldata['MSZoning'].mode()[0])
alldata = alldata.drop(['Utilities'], axis=1)
alldata["Functional"] = alldata["Functional"].fillna("Typ")
alldata['Electrical'] = alldata['Electrical'].fillna(alldata['Electrical'].mode()[0])
alldata['KitchenQual'] = alldata['KitchenQual'].fillna(alldata['KitchenQual'].mode()[0])
alldata['Exterior1st'] = alldata['Exterior1st'].fillna(alldata['Exterior1st'].mode()[0])
alldata['Exterior2nd'] = alldata['Exterior2nd'].fillna(alldata['Exterior2nd'].mode()[0])
alldata['SaleType'] = alldata['SaleType'].fillna(alldata['SaleType'].mode()[0])
alldata['MSSubClass'] = alldata['MSSubClass'].fillna("None")

# Check if there are any missing values left
alldata_na = (alldata.isnull().sum() / len(alldata)) * 100
alldata_na = alldata_na.drop(alldata_na[alldata_na == 0].index).sort_values(ascending=False)
misdata = pd.DataFrame({'Missing Ratio' :alldata_na})
misdata.head()

alldata['OverallCond'].describe()

#MSSubClass =The building class
alldata['MSSubClass'] = alldata['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
alldata['OverallCond'] = alldata['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
alldata['YrSold'] = alldata['YrSold'].astype(str)
alldata['MoSold'] = alldata['MoSold'].astype(str)

alldata['MoSold'].describe()

alldata['GarageQual'].unique() #but how will machine understand gd is better than ta but less than ex,so encoding

from sklearn.preprocessing import LabelEncoder
colms = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
         'BsmtExposure', 'GarageFinish', 'LandSlope','BsmtFinType2',  'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold','Functional', 'Fence','LotShape', 'PavedDrive', 'Street')
# Process columns and apply LabelEncoder to above categorical features
for c in colms:
    lbl = LabelEncoder() 
    lbl.fit(list(alldata[c].values)) 
    alldata[c] = lbl.transform(list(alldata[c].values))

# shape        
print('Shape alldata: {}'.format(alldata.shape))

# Adding Total Square Feet feature 
alldata['TotalSF'] = alldata['TotalBsmtSF'] + alldata['1stFlrSF'] + alldata['2ndFlrSF']
# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column to reduce skewness
sns.set_palette("BrBG", 7)
trainset["SalePrice"] = np.log1p(trainset["SalePrice"])

#The new distribution 
sns.distplot(trainset['SalePrice'] , fit=norm);


# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(trainset['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Probability density plot
fig = plt.figure()
result = stats.probplot(trainset['SalePrice'], plot=plt)
plt.show()

y_train = trainset.SalePrice.values

print("Skewness: %f" % trainset['SalePrice'].skew())
print("Kurtosis: %f" % trainset['SalePrice'].kurt())

numfeats = alldata.dtypes[alldata.dtypes != "object"].index

# Checking the skew of all numerical features
skewfeats = alldata[numfeats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skewed Features' :skewfeats})
skewness.head()

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_feats = skewness.index
lam = 0.15
for feat in skewed_feats:
    alldata[feat] = boxcox1p(alldata[feat], lam)
    alldata[feat] += 1

alldata = pd.get_dummies(alldata)
print(alldata.shape)

trainset = alldata[:n_train]
testset = alldata[n_train:]
print(trainset)
print(testset)

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Cross-validation with k-folds
nfolds = 5

def rmsle_cv(model):
    kfld = KFold(nfolds, shuffle=True, random_state=42).get_n_splits(trainset.values)
    rmse= np.sqrt(-cross_val_score(model, trainset.values, y_train, scoring="neg_mean_squared_error", cv = kfld))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
xgbmodel = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                             random_state =7)
lgbmodel = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class StackingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, basemodels, metamodel, nfolds=5):
        self.basemodels = basemodels
        self.metamodel = metamodel
        self.nfolds = nfolds
   
    # fitting the data on clones of the original models
    def fit(self, X, y):
        self.basemodels_ = [list() for x in self.basemodels]
        self.metamodel_ = clone(self.metamodel)
        kfold = KFold(n_splits=self.nfolds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions of 3 columns that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.basemodels)))
        for i, cl in enumerate(self.basemodels):
            for trainindex, houtindex in kfold.split(X, y):
                inst = clone(cl)
                self.basemodels_[i].append(inst)
                inst.fit(X[trainindex], y[trainindex])
                ypred = inst.predict(X[houtindex])
                out_of_fold_predictions[houtindex, i] = ypred
                
        # Now train the cloned  meta-model using the out-of-fold predictions
        self.metamodel_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        metafeats = np.column_stack([
            np.column_stack([model.predict(X) for model in basemodels]).mean(axis=1)
            for basemodels in self.basemodels_ ])
        return self.metamodel_.predict(metafeats)

stacked_models = StackingModels(basemodels = (ENet, KRR ,GBoost),
                                                 metamodel = lasso)
score = rmsle_cv(stacked_models) 
print("Stacking models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

stacked_models.fit(trainset.values, y_train)
stackedtrain_pred = stacked_models.predict(trainset.values)
stackedpred = np.expm1(stacked_models.predict(testset.values))#expm1 to remove log applied earlier
strmse=rmse(y_train, stackedtrain_pred)
print(strmse)

xgbmodel.fit(trainset, y_train)
xgbtrain_pred = xgbmodel.predict(trainset)
xgbpred = np.expm1(xgbmodel.predict(testset))
xgrmse=rmse(y_train, xgbtrain_pred)
print(xgrmse)

lgbmodel.fit(trainset, y_train)
lgbtrain_pred = lgbmodel.predict(trainset)
lgbpred = np.expm1(lgbmodel.predict(testset))
lgrmse=rmse(y_train, lgbtrain_pred)
print(lgrmse)

#RMSE on the entire Train data when averaging choosing random multipliers

print('RMSE score on train data:')
print(rmse(y_train,stackedtrain_pred*0.70 +
               xgbtrain_pred*0.10 + lgbtrain_pred*0.20 ))

#weights using the rmse
stack=1/strmse
xgb=1/xgrmse
lgb=1/lgrmse
sum=stack+xgb+lgb
st=stack/sum
xg=xgb/sum
lg=lgb/sum
print(st,xg,lg)

#Weighted average using rmse
print('RMSE score on train data:')
print(rmse(y_train,stackedtrain_pred*st +
               xgbtrain_pred*xg + lgbtrain_pred*lg))

ensemble1 = stackedpred*st + xgbpred*xg + lgbpred*lg


pred1 = pd.DataFrame()
pred1['Id'] = test_ID
pred1['SalePrice'] = ensemble1
pred1.to_csv('pred1.csv',index=False)