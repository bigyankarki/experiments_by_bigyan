# importing dependencies
import tensorflow as tf
import numpy as np
from api.digit_recognition.process_image import processed_image

def return_prediction(img):

    tf.reset_default_graph()
    # ----------------------------- CONSTRUCTION PHASE ------------------

    # Defining parameters.
    n_inputs = 28*28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10  # 10 Classes of prediction

    # defining placeholder variables.
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.int32, shape=(None), name='y')

    # defining deep neural network.
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(x, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


    # defining loss function.
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    # defining the neural network optimizer: the graident descent.
    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    # measuring classifiers acuracy.
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # initializing all variables.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    # ------------------------EXECUTION PHASE--------------------------
    with tf.Session() as sess:
        saver.restore(sess, './api/digit_recognition/checkpoints/my_model_final.ckpt')
        x_new_scaled = [processed_image(img)]
        z = logits.eval(feed_dict={x: x_new_scaled})
        y_pred = np.argmax(z, axis=1)[0]

        y_conf = tf.nn.softmax(z, 1)
        y_conf_list = [("%.3f" % i) for i in y_conf.eval(feed_dict={x: x_new_scaled})[0]]

    return y_pred, y_conf_list[y_pred]

if __name__ == "__main__":
    return_prediction(base64_img)
