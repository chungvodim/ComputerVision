#The central unit of data in TensorFlow is the tensor
#TensorFlow Core programs as consisting of two discrete sections:
#    Building the computational graph.
#    Running the computational graph.

import tensorflow as tf

#To actually evaluate the nodes, we must run the computational graph within a session
with tf.Session() as sess:
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1, node2)
    print(sess.run([node1, node2]))

    #We can build more complicated computations by combining Tensor nodes with operations (Operations are also nodes.)
    node3 = tf.add(node1, node2)
    print("node3: ", node3)
    print("sess.run(node3): ",sess.run(node3))

    #A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later.
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)

    #We can evaluate this graph with multiple inputs by using the feed_dict parameter to specify Tensors that provide concrete values to these placeholders:
    print(sess.run(adder_node, {a: 3, b:4.5}))
    print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))#print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))

    #We can make the computational graph more complex by adding another operation. For example,
    add_and_triple = adder_node * 3.
    print(sess.run(add_and_triple, {a: 3, b:4.5}))

    #To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. Variables allow us to add trainable parameters to a graph
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    #To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
    init = tf.global_variables_initializer()
    sess.run(init)

    #Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
    print(sess.run(linear_model, {x:[1,2,3,4]}))

    #A loss function measures how far apart the current model is from the provided data
    # sum all the squared errors to create a single scalar that abstracts the error
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

    #We could improve this manually by reassigning the values of W and b to the perfect values of -1 and 1
    #fixW = tf.assign(W, [-1.])
    #fixb = tf.assign(b, [1.])
    #sess.run([fixW, fixb])
    #print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

    #TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest optimizer is gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess.run(init) # reset values to incorrect defaults.
    for i in range(1000):
      sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

    print(sess.run([W, b]))

    # evaluate training accuracy
    # training data
    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#Destroy session