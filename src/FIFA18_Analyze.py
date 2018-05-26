
# coding: utf-8

# <br><br><br>
# <center>
# <b><font size="+3">CS584: FIFA18 data Analysis </font></b>
# <br><br><br><br>
# *Ting Jiang*<br>
# *Chen Gong*<br>
# *Yizhi Hong*
# </center>
# <br><br><br>

# **** <h3>Part 1: Data pre-processing</h3> ****
# 

# In[1]:


import numpy as np
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
sns.set_style("dark")


# **** <h4>look up data firt 10 columns</h4> ****

# In[3]:


dataframe = pd.read_csv('../data/fifa-18-demo-player-dataset/CompleteDataset.csv')
dataframe.head(10)


# In[4]:


dataframe.columns


# **** take the attribute that we need to use ****
# <p>The attribute needs to be predicted: 'Overall','Preferred Positions'</p>
# <p>The attribute use to predict: rest of the attributes</p>
#        

# In[5]:


# only consider non goalkeeper's position.

col_needed = ['Overall','Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys', 'Preferred Positions']

# rearrange the attributes. The attribute need to be predicted: Overall, Preferred Position
# rearrange as ST -> CM -> CB

recol_needed = ['Overall','Finishing', 'Shot power', 'Positioning', 'Dribbling', 'Long shots','Penalties', 'Volleys', 
                 'Acceleration', 'Agility','Sprint speed', 'Curve',
                
       'Free kick accuracy', 'Heading accuracy', 
       'Short passing', 'Long passing', 'Vision',
       'Strength', 'Stamina', 'Balance', 'Ball control','Composure','Jumping', 
       'Crossing','Reactions',
       'Aggression','Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle','Preferred Positions']

dataframe = dataframe[recol_needed]
dataframe.head(10)


# In[6]:


dataframe['Preferred Positions'] = dataframe['Preferred Positions'].str.strip()
#remove Goalkeeper from dataframe

dataframe = dataframe[dataframe['Preferred Positions'] != 'GK']
dataframe.head(10)


# **** Check the data  ****

# In[7]:


# make sure no null value.
dataframe.isnull().values.any()


# In[8]:


# Check all the positions we have.
positions = dataframe['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()
positions


# In[9]:


# handle multiple positions
df_fifa = dataframe.copy()
df_fifa.drop(df_fifa.index, inplace=True)

for position in positions:
    temp = dataframe[dataframe['Preferred Positions'].str.contains(position)]
    temp['Preferred Positions'] = position
    df_fifa = df_fifa.append(temp, ignore_index=True)
    
df_fifa.iloc[::1000, :]
            


# In[10]:


cols = [col for col in df_fifa.columns if col not in ['Preferred Positions']]

for i in cols:
    df_fifa[i] = df_fifa[i].apply(lambda x: eval(x) if isinstance(x,str) else x)

df_fifa.iloc[::1000, :]


# **** <h3>Part2: Data Analyze </h3> ****
# **** <h4>The plot below shows how the attributes contribute the position. </h4> ****

# In[11]:


fig, fs = plt.subplots()

## show the 3 main positions  
df_ST = df_fifa[df_fifa['Preferred Positions'] == 'ST'].iloc[::10,:-1]
np.mean(df_ST).T.plot.line(color = 'red', figsize = (15,10), legend = 'ST',label='ST', ylim = (0, 110), title = "attributes distribution", ax=fs)

df_CM = df_fifa[df_fifa['Preferred Positions'] == 'CM'].iloc[::10,:-1]
np.mean(df_CM).T.plot.line(color = 'blue', figsize = (15,10), legend = 'CM',label='CM', ylim = (0, 110), title = "attributes distribution", ax=fs)

df_CB = df_fifa[df_fifa['Preferred Positions'] == 'CB'].iloc[::10,:-1]
np.mean(df_CB).T.plot.line(color = 'green', figsize = (15,10), legend = 'CB',label='CB', ylim = (0, 110), title = "attributes distribution", ax=fs)



fs.set_xlabel('Attributes')
fs.set_ylabel('Rating')

fs.set_xticks(np.arange(len(cols)))
fs.set_xticklabels(labels = cols, rotation=90)

for l in fs.lines:
    l.set_linewidth(1)

fs.axvline(0, color='red', linestyle='--')   
fs.axvline(12, color='red', linestyle='--')

fs.axvline(12.1, color='blue', linestyle='--')
fs.axvline(24, color='blue', linestyle='--')

fs.axvline(24.1, color='green', linestyle='--')
fs.axvline(29, color='green', linestyle='--')

fs.text(4, 85, 'Attack Attributes', color = 'red', weight = 'bold')
fs.text(15.5, 85, 'Mixed Attributes', color = 'blue', weight = 'bold')
fs.text(25, 85, 'Defend Attributes', color = 'green', weight = 'bold')
plt.show()


# **** we can see above there is obvious margin between attacker's attributes and defender's attributes  ****

# *** <h3>1. Logistic Regression </h3> ***
# 
# *****  predict the Attacker or the Defender  *****

# **** Set the ST/RW/LW/RM/CM/LM/CAM/CF as an Attacker group --> 1 ****
# 
# **** Set the CDM/CB/LB/RB/RWB/LWB as an Defender group --> 0 ****

# In[12]:


# Set the baseline of the prediction
baseline = 1/2
print('The baseline is', baseline)


# In[13]:


df_fifa_normalized = df_fifa.iloc[:,:-1].div(df_fifa.iloc[:,:-1].sum(axis=1), axis=0)
mapping = {'ST': 1, 'RW': 1, 'LW': 1, 'RM': 1, 'CM': 1, 'LM': 1, 'CAM': 1, 'CF': 1, 'CDM': 0, 'CB': 0, 'LB': 0, 'RB': 0, 'RWB': 0, 'LWB': 0}

df_fifa_normalized['Preferred Positions'] = df_fifa['Preferred Positions']

df_fifa_normalized = df_fifa_normalized.replace({'Preferred Positions': mapping})
df_fifa_normalized.iloc[::1000,]


# In[14]:


# perform 5 cross validation
clf = LogisticRegression()
x = df_fifa_normalized.iloc[:,:-1]
y = df_fifa_normalized.iloc[:,-1]
scores = cross_val_score(clf, x, y, cv=5)
print ('Logistic Regression Accuracy: {}'.format(np.mean(scores)))


# **** Tune the features by lasso ****

# In[15]:


# Perform lasso to get rid of the attribute that unnecessary influence the decision of position
clf = Lasso(alpha=0.00001)
clf.fit(x,y)
Feature_Coef_list = list(sorted(zip(recol_needed, abs(clf.coef_)),key=lambda x: -x[1]))
Feature_Coef_table = pd.DataFrame(np.array(Feature_Coef_list).reshape(-1,2), columns = ['Attributes', 'Coefficient'])
print(Feature_Coef_table)


# **** now we try to enumerate the features to get the highest performance ****

# In[16]:


max_score = 0
n_features = 0

for i in range(1,len(Feature_Coef_table['Attributes'])):
    clf_lasso = LogisticRegression()
    lasso_cols = Feature_Coef_table[:i]['Attributes'].tolist()
    x_lasso = df_fifa_normalized.iloc[:,:-1][lasso_cols]
    scores_lasso = cross_val_score(clf_lasso, x_lasso,y , cv=5)
    if np.mean(scores_lasso) > max_score:
        max_score = np.mean(scores_lasso)
        n_features = i

print ('Logistic Regression Accuracy (' + str(n_features) +' features):' + str(max_score))


# **** As we can see here. we are improve the accuracy slightly ****
# 
# **** And it is higher than baseline 0.5 ****

# In[17]:


imp_features = Feature_Coef_table[:n_features]['Attributes'].tolist()
print('The important features to determine the 1/0 is')
print(imp_features)


# *** <h3>2. Random Forest</h3>  ***
# 
# *****  predict all the position *****

# In[18]:


# Set the baseline of the prediction
baseline = 1/len(positions)
print('The baseline is', baseline)


# In[19]:


df_fifa_all_pos = df_fifa.copy()
mapping_all = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CF': 7, 'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}

df_fifa_all_pos = df_fifa_all_pos.replace({'Preferred Positions': mapping_all})
df_fifa_all_pos.iloc[::1000,]


# In[20]:


# perform 5 cross validation
clf = LogisticRegression()
x = df_fifa_all_pos.iloc[:,:-1]
y = df_fifa_all_pos.iloc[:,-1]
log_scores = cross_val_score(clf, x, y, cv=3)
print ('Logistic Regression Accuracy: {}'.format(np.mean(log_scores)))


# In[21]:


clf = RandomForestClassifier(random_state=0)
x = df_fifa_all_pos.iloc[:,:-1]
y = df_fifa_all_pos.iloc[:,-1]
rf_scores = cross_val_score(clf, x, y, cv=3)
print ('Random Forest Accuracy: {}'.format(np.mean(rf_scores)))


# **** Tune the features by ridge ****

# In[22]:


# Perform ridge to get the importance of the feature when determining the position.
clf = Ridge(alpha=0.001)
clf.fit(x,y)
Feature_Coef_list = list(sorted(zip(recol_needed, abs(clf.coef_)),key=lambda x: -x[1]))
Feature_Coef_table = pd.DataFrame(np.array(Feature_Coef_list).reshape(-1,2), columns = ['Attributes', 'Coefficient'])
print(Feature_Coef_table)


# **** now we try to enumerate the features to get the highest performance ****

# In[23]:


max_score = 0
n_features = 0

for i in range(1,len(Feature_Coef_table['Attributes'])):
    clf_ridge = RandomForestClassifier(random_state=0)
    ridge_cols = Feature_Coef_table[:i]['Attributes'].tolist()
    x_ridge = df_fifa_normalized.iloc[:,:-1][ridge_cols]
    scores_ridge = cross_val_score(clf_ridge, x_ridge,y , cv=3)
    if np.mean(scores_ridge) > max_score:
        max_score = np.mean(scores_ridge)
        n_features = i

print ('Random Forest Accuracy (' + str(n_features) +' features):' + str(max_score)) 


# In[24]:


imp_features = Feature_Coef_table[:n_features]['Attributes'].tolist()
print('The important features to determine the positon is')
print(imp_features)


# **** As we can see here. we are improve the accuracy slightly ****
# 
# **** And 0.395765861155 is higher than baseline 0.07142857142857142 ****

# *** <h3>3. Linear Regression</h3> ***
# ***** predict the overall of the player. *****

# **** define a new cross validation ****

# In[25]:


def cross_Validation_reg(reg, X, y, k = 3):
    
    tMSE = list()

    for train_index, test_index in KFold(n_splits=k, random_state=None, shuffle=False).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regm = reg.fit(X_train, y_train)
        tMSE.append(np.mean((y_test - regm.predict(X_test)) ** 2))
    return np.mean(tMSE)


# In[26]:


## set y overall
overall = np.array(df_fifa.iloc[:,0:1])[:,0]
Xb = csr_matrix(df_fifa.iloc[:, 1:-1])
Xb.toarray()


# In[27]:


class baseline:
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        n = X.shape[0]
        res = np.zeros(n)
        for i in range(n):
            res[i] = np.mean(X[i,:])
        return res

# set the baseline class for certain player
bl = baseline()


# In[28]:


# test baseline for all
cross_Validation_reg(bl, Xb, overall, 5)


# **** Perform linear model ****

# In[29]:


overall = np.array(df_fifa.iloc[:,0:1])[:,0]
X = csr_matrix(df_fifa_all_pos.iloc[:, :])
lr = LinearRegression()

## ignore the positions
accuracy = cross_Validation_reg(lr, Xb,overall, 5)
print('The linear model Accuracy(ignore the positions):' + str(accuracy))

lr = LinearRegression()
## fatorize the positions
accuracy_f = cross_Validation_reg(lr, X, overall, 5)
print('The linear model Accuracy(fatorize the positions):' + str(accuracy_f))


# In[30]:


lrm = lr.fit(X, overall)
print('The coef are ' + str(lrm.coef_))
print('The intercept is ' + str(lrm.intercept_))


# **** Perform polynomial model ****

# In[31]:


model = make_pipeline(PolynomialFeatures(2), Ridge(copy_X = False))
accuracy_p = cross_Validation_reg(model, X.toarray(), overall, 5)
print('The polyomial model accuracy (factorize the positions):' + str(accuracy))


# In[32]:


nX = X.toarray()
modelp = model.fit(nX , overall)
result = modelp.predict(nX)
print('To predict all the player overall rating by our model')
print(result)


# **** we get a very good accuracy in polynomial model  ****
