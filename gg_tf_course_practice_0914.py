from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow.compat.v1 as tf
import tensorflow
from tensorflow.python.data import Dataset
import sys


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# load data

california_housing_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

print (len(california_housing_df))

california_housing_df = california_housing_df.reindex(
                        np.random.permutation(california_housing_df.index)
                        )

# print (california_housing_df.columns)

# print (list(california_housing_df.median_house_value))

# /= += -= these operations can be directly applied to a value column
# scale the numbers to make it easier to train in usual learning rates
california_housing_df['median_house_value'] /= 1000.0

# print (list(california_housing_df.median_house_value))

# describe function to show some key statistics of the value columns
print (california_housing_df.describe())


# define the input feature, this one has the real data
my_feature = california_housing_df[['total_rooms']]

# configure a numeric feature column for total_rooms, this doesn't contain any real datasets
# but just a description
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# define the label, this one has the real data
# label is the prediction we want the model to output
# it is also the results that training dataset has
# simlified as 'The Y'
targets = california_housing_df['median_house_value']

# use grdient descent as the optimizer for training the model
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# gradient clipping ensures the magnitude of the gradients do not become too large during training,
# which can cause gradient descent to fail

my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)


# configure the linear regression model with our feature columns and my_optimizer
# set a learning rate of 0.0000001 for gradient descent
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays
    features = {key:np.array(value) for key, value in dict(features).items()}


    # construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


_ = linear_regressor.train(
    input_fn = lambda: my_input_fn(my_feature, targets), steps=100
)


# Create an input function for prediections
# note: Since we're making just one prediction each example, we don't
# need to repeat or shuffle the data here
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# format predictions as a NumPy array, so we can calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

# print mean squared error and root mean squared error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

print ("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print ("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
















# #
