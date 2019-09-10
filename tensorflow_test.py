import tensorflow as tf
import numpy as np



# sample 1, predict a linear relationship
# step 1: create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# weight = 0.1, bias = 0.3

# step 2: create tensorflow structure start

# set up the initial value
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# y is the prediction, we want to increase the accuracy of this prediction
y = Weights * x_data + biases

# the cost function of the prediction to the real value
loss = tf.reduce_mean(tf.square(y-y_data))

# gradient descent optimizer, the parameter is the efficiency, <1 value
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()

# structure end

# step 3, start a session to train the neural network
sess = tf.compat.v1.Session()
sess.run(init) # very important, needs to be activated

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(Weights), sess.run(biases))
