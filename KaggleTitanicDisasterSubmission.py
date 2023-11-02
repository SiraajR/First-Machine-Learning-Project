from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

DStrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
DStest =  pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = DStrain.pop('survived')
y_test = DStest.pop('survived')
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = DStrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def form_input_function(dataSet, labelSet, numOfEpochs= 10, shuffle = True, batch_size = 32 ):
  def input_function():
    Data = tf.data.Dataset.from_tensor_slices((dict(dataSet), labelSet))
    if shuffle:
      Data = Data.shuffle(1000)
    Data = Data.batch(batch_size).repeat(numOfEpochs)
    return Data
  return input_function

train_input_fn = form_input_function(DStrain, y_train)
test_input = form_input_function(DStest, y_test,numOfEpochs= 1 , shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(test_input)
clear_output()
print(result['accuracy'])

pred_dict = list(linear_est.predict(test_input))
prob = pd.Series([pred['probabilities'][1] for pred in pred_dict])
prob.plot(kind ='hist', bins = 20 , title = 'predicted probability')
