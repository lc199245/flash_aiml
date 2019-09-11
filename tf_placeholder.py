import tensorflow.compat.v1 as tf


# define variables that are placeholders
input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)


# mul has been renamed to multiply
# plus the sub function has been renamed to subtract
output = tf.multiply(input_1, input_2)

with tf.Session() as sess:
    print (sess.run(output, feed_dict={input_1:[7.], input_2:[2.]  } ))
