# two ways of opening a session




import tensorflow as tf





matrix_1 = tf.constant([[3,3]])
matrix_2 = tf.constant([[2],
                        [2]])


# matrix multiply similar to np.dot(m1,m2)
product = tf.matmul(matrix_1, matrix_2)


# print (product)


# method 1
sess = tf.Session()
result = sess.run(product)
print (result)

sess.close()


# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print (result2)
