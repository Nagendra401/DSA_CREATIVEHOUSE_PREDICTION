import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10.0, 8.0)
from scipy import stats
from scipy.stats import norm
#from fancyimpute import MICE

data = pd.read_csv("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\DSA_HACKTHON_1\\1522419498_DSA_Hackathon_Dataset.csv")
data.head()
data.columns

validation_data = pd.read_csv("C:\\Users\\NAGENDRA_CHAPALA\\Documents\\7400618_810833986\\DataScienceCourse_TechEssential\\course\\DATA_SCIENCE_AUTHORITY\\DSA_HACKTHON_1\\1522419497_DSA_Hackathon_Validation_Dataset.csv")
validation_data.head()
validation_data.columns


########### Data preparation #####################

data.describe()
data.info()

print('The train data has {0} rows and {1} columns'.format(data.shape[0],data.shape[1]))
print('----------------------------')
print('The test data has {0} rows and {1} columns'.format(validation_data.shape[0],validation_data.shape[1]))

#check missing values
data.columns[data.isnull().any()]
#Out of 16 features, 14 features have missing values. Let's check the percentage of missing values in these columns.

#missing value counts in each of these columns
miss = data.isnull().sum()/len(data)
miss
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss
type(miss)

#visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index
miss

#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
plt.show()


####### data visualization ################

#check the distribution of the target variable
sns.distplot(data['price'])

#need to log transform this variable so that it becomes normally distributed
#skewness
print("The skewness of SalePrice is {}".format(data['price'].skew()))

#now transforming the target variable
target = np.log(data['price'])
print("skewness is: ", target.skew())
sns.distplot(target)

#separate variables into new data frames
numeric_data = data.select_dtypes(include=[np.number])
cat_data = data.select_dtypes(exclude=[np.number])
print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))

del data['id']
data.columns

#create numeric plots
num = [f for f in data.columns if f != 'price']
num
nd = pd.melt(data, value_vars = num)
nd
n1 = sns.FacetGrid(nd, col='variable', col_wrap=4, sharex=False, sharey=False)
type(n1)
n1 = n1.map(sns.distplot, 'value', kde_kws={"color": "k", "lw": 3, "label": "KDE"})
n1


def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)
cat = [f for f in data.columns if data.dtypes[f] != 'object']
p = pd.melt(data, id_vars='price', value_vars=cat)
g = sns.FacetGrid(p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value', 'price')
g

#create box plots
sns.boxplot(x="variable", y="value", data=nd)
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=5)
for i, column in enumerate(data.columns):
    ax = axes[i // 3, i % 3]
    sns.distplot(data[column], ax=ax)
    ax.set_ylim(0, 0.09)


#correlation plot
corr = data.corr()
sns.heatmap(corr)

print(corr['price'].sort_values(ascending=False), '\n')
print("--------------------------------------")
print(corr['price'].sort_values(ascending=False)[-5:], '\n')

#sqft_living plot
sns.jointplot(x=data['sqft_living'], y=data['price'])
max(data['price'])
max(data['sqft_living'])

################ Data Pre-Processing ####################################
data.shape

data1 = data.dropna(thresh=10)
data1.shape

#removing outliers
data2 = data1.drop(data1[data1['sqft_living'] > 12000].index)
data2.shape #removed 2 rows

data2.columns[data2.isnull().any()]

data2.describe()

#imputing missing values
data2["bedrooms"].fillna(0, inplace=True)
data2["sqft_lot"].fillna(0, inplace=True)

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = data2.apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
data2[skewed] = np.log1p(data2[skewed])

# target = data2['price']
# target.shape

#create a label set
label_df = pd.DataFrame(index = data2.index, columns = ['price'])
label_df['price'] = np.log(data2['price'])
label_df.shape
data2.shape

del data2['price']
data2.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data2)
data2 = scaler.transform(data2)

scaler.fit(label_df)
label_df = scaler.transform(label_df)

label_df.shape
data2.shape

######################## models building ########################
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(data2, label_df, test_size = 0.20, random_state = 2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)


############# GradientBoostingRegressor ###########################
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')

clf.fit(x_train, y_train.ravel())
clf.score(x_test,y_test)

#################### XGBRegressor ########################
import xgboost as xgb
from sklearn.metrics import mean_squared_error
regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

regr.fit(x_train, y_train)
preds = regr.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

######################### Lasso ######################
from sklearn.linear_model import Lasso
#found this best alpha through cross-validation
best_alpha = 0.00099
regr_lasso = Lasso(alpha=best_alpha, max_iter=50000)
regr_lasso.fit(x_train, y_train)
y_pred = regr_lasso.predict(x_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (rmse_lasso))



############ Keras ###################
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

np.random.seed(10)

#create Model
#define base model
def base_model():
     model = Sequential()
     model.add(Dense(20, input_dim=398, init='normal', activation='relu'))
     model.add(Dense(10, init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     model.compile(loss='mean_squared_error', optimizer = 'adam')
     return model

seed = 7
np.random.seed(seed)
clf = KerasRegressor(build_fn=base_model, nb_epoch=1000, batch_size=5,verbose=0)
clf.fit(x_train,y_train)
#make predictions and create the submission file
kpred = clf.predict(x_test)
kpred = np.exp(kpred)
pred_df = pd.DataFrame(kpred, columns=["price"])
pred_df.to_csv('keras1.csv', header=True, index_label='Id')