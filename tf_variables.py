import tensorflow.compat.v1 as tf



# define a variable
state = tf.Variable(0, name='counter')

# print (state.name)

# define a new constant
one = tf.constant(1)

new_value = tf.add(state, one)

update = tf.assign(state, new_value)

# must initial all defined variables and constants
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(3):
        sess.run(update)
        # must 'run' to check the value
        print (sess.run(state))
