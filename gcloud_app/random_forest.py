#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
from sklearn import tree
import pandas as pd
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt


# In[2]:


#import our train and test csv
train_path = os.path.join("Data/train.csv")
train_df = pd.read_csv(train_path)

test_path = os.path.join("Data/test.csv")
test_df = pd.read_csv(test_path)


# In[3]:


#drop na in test_df
test_df = test_df.dropna()


# In[4]:


#drop na in train_df
train_df = train_df.dropna()


# In[5]:


#convert the satisfaction column to 1's and 0's in the train_df
train_df["satisfaction"].replace({"satisfied":1, "neutral or dissatisfied":0}, inplace=True)


# In[6]:


#convert gender column to 1's and 0's in train_df
train_df["Gender"].replace({"Male":1, "Female":0}, inplace=True)


# In[7]:


#convert customer type column to 1's and 0's in train_df
train_df["Customer Type"].replace({"Loyal Customer":1, "disloyal Customer":0}, inplace=True)


# In[8]:


#convert travel type column to 1's and 0's in train_df
train_df["Type of Travel"].replace({"Business travel":1, "Personal Travel":0}, inplace=True)


# In[9]:


#convert class type column to 2's 1's and 0's in train_df
train_df["Class"].replace({"Business":2, "Eco":1, "Eco Plus":0}, inplace=True)


# In[10]:


#assign the X_train data
train_df = train_df.drop(columns=["Unnamed: 0", 'id'])

#assign the y_train data
y_train = np.array(train_df["satisfaction"]).reshape(-1,1)


# In[11]:


#assign x_train values
X_train = train_df


# In[12]:


#drop target variable from x train
X_train = X_train.drop(columns=["satisfaction"])


# In[13]:


#convert the satisfaction column to 1's and 0's in test_df
test_df["satisfaction"].replace({"satisfied":1, "neutral or dissatisfied":0}, inplace=True)


# In[14]:


#convert gender column to 1's and 0's in test_df
test_df["Gender"].replace({"Male":1, "Female":0}, inplace=True)


# In[15]:


#convert customer type column to 1's and 0's in test_df
test_df["Customer Type"].replace({"Loyal Customer":1, "disloyal Customer":0}, inplace=True)


# In[16]:


#convert travel type column to 1's and 0's in test_df
test_df["Type of Travel"].replace({"Business travel":1, "Personal Travel":0}, inplace=True)


# In[17]:


#convert class type column to 2's 1's and 0's in test_df
test_df["Class"].replace({"Business":2, "Eco":1, "Eco Plus":0}, inplace=True)


# In[18]:


#drop unneeded columns and reshape y_test variable
test_df = test_df.drop(columns=["Unnamed: 0", 'id'])

y_test = np.array(test_df["satisfaction"]).reshape(-1,1)


# In[19]:


#assign x_test variables
X_test = test_df


# In[20]:


#drop the target variable from the X_test variables
X_test = X_test.drop(columns=["satisfaction"])


# In[21]:


#establish the feature names for random forest importance metrics
feature_names = X_test.columns


# In[22]:


#create model using random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, oob_score=True)
rf = rf.fit(X_train, y_train)


# In[26]:


#score the model using oob score
oob_score = rf.oob_score_


# In[27]:


#score the model at 200 estimators
model_score = rf.score(X_test, y_test)


# In[32]:


#assign n_estimators to a variable for table
estimators = rf.n_estimators


# In[ ]:


#print out the feature importances
sorted(zip(rf.feature_importances_, feature_names), reverse=True)


# In[ ]:


#save model using pickle
filename = 'gcloud_app/random_forest.pkl'
pickle.dump(rf, open(filename, 'wb'))


# In[ ]:


import matplotlib.pyplot as plt

# from collections import OrderedDict
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier

# Author: Kian Ho <hui.kian.ho@gmail.com>
#         Gilles Louppe <g.louppe@gmail.com>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#
# License: BSD 3 Clause

# print(__doc__)

# RANDOM_STATE = 123

# # Generate a binary classification dataset.
# X, y = make_classification(n_samples=500, n_features=25,
#                            n_clusters_per_class=1, n_informative=15,
#                            random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
# ensemble_clfs = [
#     ("RandomForestClassifier, max_features='sqrt'",
#         RandomForestClassifier(warm_start=True, oob_score=True,
#                                max_features="sqrt",
#                                random_state=RANDOM_STATE)),
#     ("RandomForestClassifier, max_features='log2'",
#         RandomForestClassifier(warm_start=True, max_features='log2',
#                                oob_score=True,
#                                random_state=RANDOM_STATE)),
#     ("RandomForestClassifier, max_features=None",
#         RandomForestClassifier(warm_start=True, max_features=None,
#                                oob_score=True,
#                                random_state=RANDOM_STATE))
# ]

# # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 10
max_estimators = 500

x_values = np.arange(min_estimators, max_estimators + 1, 50).tolist()
y_values = []

for i in range(min_estimators, max_estimators + 1, 50):
    model = RandomForestClassifier(n_estimators=i, oob_score=True).fit(X_train, y_train)
    y_values.append(model.oob_score_)


# In[ ]:


plt.plot(x_values, y_values)
plt.xlabel("n_estimators", fontweight='heavy')
plt.ylabel("oob_score", fontweight='heavy')
plt.title("oob_score vs. n_estimators", fontweight='bold')
plt.savefig("images/oob_score_vs_estimators", bbox_inches="tight")


# In[ ]:


#assign importances to variable
importances = rf.feature_importances_ 


# In[ ]:


#get standard deviation of importances
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)


# In[ ]:


#convert importances to pandas series
forest_importances = pd.Series(importances, index=feature_names)


# In[ ]:


#plot feature importances and standard deviation as yerr bar
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature Importances Using MDI", weight='bold')
ax.set_ylabel("Mean Decrease in Impurity", weight='bold')
plt.savefig("images/feature_importance", bbox_inches="tight")


# In[49]:


table = pd.DataFrame([{"Model Type": "Random Forest", "# Estimators": estimators, "Model Score": model_score, "OOB Score": oob_score}])
table


# In[45]:


table = pd.DataFrame.transpose(table)


# In[46]:


table


# In[51]:


html_table = table.to_html(index=True, header=False)


# In[52]:


html_table


# In[ ]:




