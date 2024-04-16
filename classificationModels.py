import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sn
import tensorflow as tf
import random
import torch

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib.pyplot import figure
from numpy import mean
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

# Load test and train datasets

train_data = pd.read_csv("Analysis1_training.csv")
test_data = pd.read_csv("Analysis1_test.csv")

# Remove response variable from datasets and make label variable

train_features = train_data.drop(['cluster_index'],axis =1)
train_labels = train_data['cluster_index']
test_features = test_data.drop(['cluster_index'],axis =1)
test_labels  = test_data['cluster_index']

x_train = train_features.values
y_train = train_labels.values
x_test = test_features.values
y_test = test_labels.values


############## Supervised learning models ################################

#### LightGBM #####

import lightgbm as lgb
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

#Train the model
parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 30,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 31,
    'learning_rate': 0.05,
    'verbose': 0,
    'seed': 42
}

model_lgb = lgb.train(parameters, lgb_train,valid_sets=lgb_eval,num_boost_round=500)
y_pred = model_lgb.predict(x_test)

y_pred_lgb_model = model_lgb.predict(np.array(x_test))
fpr_lgb_model, tpr_lgb_model, thresholds_lgb_model = roc_curve(np.array(y_test), y_pred_lgb_model)
auc_lgb_model = auc(fpr_lgb_model, tpr_lgb_model)

precision_lgb, recall_lgb , _ = precision_recall_curve(y_test, y_pred_lgb_model)
precision_recall_auc_lgb = auc(recall_lgb, precision_lgb)

#### Neural Network #####

tf.random.set_seed(42)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(45,)),
  tf.keras.layers.Dense(55, activation='relu'),
  tf.keras.layers.Dense(40, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2,activation = 'softmax')
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='sgd',
              loss=loss_fn,
              metrics=['accuracy'])
              
model.save_weights('model.h5')

x_train_np = np.array(x_train)
y_train_np = np.array(y_train)
model.fit(x_train, y_train, epochs=50)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

y_pred_nn_model = probability_model(np.array(x_test))

y_pred_nn_model = probability_model(np.array(x_test))[:, 1]
fpr_nn_model, tpr_nn_model, thresholds_nn_model = roc_curve(y_test, y_pred_nn_model)
auc_nn_model = auc(fpr_nn_model, tpr_nn_model)

precision_nn, recall_nn , _ = precision_recall_curve(y_test, y_pred_nn_model)
precision_recall_auc_nn = auc(recall_nn, precision_nn)

#### TabNet ####

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# Ensure that CUDA operations are deterministic (if using CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


train_features_deep, test_features_deep, train_labels_deep, test_labels_deep = train_test_split(x_train, y_train, test_size = 0.2, random_state = 41)
clf = TabNetClassifier(n_steps = 3)  #TabNetRegressor()

clf.fit(
  train_features_deep, train_labels_deep,
  eval_set=[(test_features_deep, test_labels_deep)],
  weights = 1
)
tab_prediction = clf.predict(test_features_deep)
tab_prediction

y_pred_keras = clf.predict_proba(x_test)
y_pred_keras = y_pred_keras[:, 1]
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)

precision_tab, recall_tab , _ = precision_recall_curve(y_test, y_pred_keras)
precision_recall_auc_tab = auc(recall_tab, precision_tab)

#### Balanced logistic regression #####

bbc_log = LogisticRegression(solver='lbfgs', class_weight='balanced',  random_state=42)
bbc_log_model = bbc_log.fit(x_train, y_train)
bbc_log_prediction = bbc_log_model.predict(x_test)

y_pred_bbc_log_model = bbc_log_model.predict_proba(x_test)[:, 1]
fpr_bbc_log_model, tpr_bbc_log_model, thresholds_bbc_log_model = roc_curve(y_test, y_pred_bbc_log_model)
auc_bbc_log_model = auc(fpr_bbc_log_model, tpr_bbc_log_model)
precision_log, recall_log , _ = precision_recall_curve(y_test, y_pred_bbc_log_model)

sorted_indices = np.argsort(recall_log) # Sort by recall (x-axis), in case they are not sorted
sorted_recall = recall_log[sorted_indices]
sorted_precision = precision_log[sorted_indices]
unique_recall, unique_indices = np.unique(sorted_recall, return_index=True)
unique_precision = sorted_precision[unique_indices]
precision_recall_auc_log = auc(unique_recall, unique_precision)

#### Balanced bagging #####

bbc_nm = BalancedBaggingClassifier(random_state=42, sampler=(SMOTE()))
bbc_nm_model = bbc_nm.fit(x_train, y_train)
bbc_nm_prediction = bbc_nm_model.predict(x_test)

y_pred_bbc_nm_model = bbc_nm_model.predict_proba(x_test)[:, 1]
fpr_bbc_nm_model, tpr_bbc_nm_model, thresholds_bbc_nm_model = roc_curve(y_test, y_pred_bbc_nm_model)
auc_bbc_nm_model = auc(fpr_bbc_nm_model, tpr_bbc_nm_model)
precision_bbc, recall_bbc , _ = precision_recall_curve(y_test, y_pred_bbc_nm_model)
precision_recall_auc_bbc = auc(recall_bbc, precision_bbc)

#### Balanced random forest #####

brf = BalancedRandomForestClassifier(random_state=42)
brf_model = brf.fit(x_train, y_train)
brf_prediction = brf_model.predict(x_test)

y_pred_rf = brf.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)
precision_rf, recall_rf , _ = precision_recall_curve(y_test, y_pred_rf)
precision_recall_auc_rf = auc(recall_rf, precision_rf)


############## Visualization of ROC curves ################################

plt.rcParams["figure.figsize"] = (8,8)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Balanced Random Forest (area = {:.2f})'.format(auc_rf))
plt.plot(fpr_bbc_nm_model, tpr_bbc_nm_model, label='Balanced Bagging (area = {:.2f})'.format(auc_bbc_nm_model))
plt.plot(fpr_bbc_log_model, tpr_bbc_log_model, label='Balanced Logistic Regression (area = {:.2f})'.format(auc_bbc_log_model))
plt.plot(fpr_lgb_model, tpr_lgb_model, label='LightGBM (area = {:.2f})'.format(auc_lgb_model))
plt.plot(fpr_keras, tpr_keras, label='TabNet (area = {:.2f})'.format(auc_keras))
plt.plot(fpr_nn_model, tpr_nn_model, label='Neural Network (area = {:.2f})'.format(auc_nn_model))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.legend(loc='best')
plt.grid(True)
plt.show()


############### Partial dependence plots for balanced random forest ########

from sklearn.inspection import PartialDependenceDisplay, partial_dependence
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
features=[12, 7, 27, 13, 42]
display = PartialDependenceDisplay.from_estimator(brf, train_features, features,line_kw={"color": "blue"})

