#===MNIST Model===
#Author: Nicolai Kozel
#Description: Convolutional neural network tensorflow model 
#to classify MNIST images. 

#Imports
#=======
#Import Tensorflow library
import tensorflow as tf
#Import MNIST dataset 
#--Handwritten digits from 0-9 (28x28px)
#--Labels expressed as one-hot vectors
from tensorflow.examples.tutorials.mnist import input_data
#Import freeze graph tool
from tensorflow.python.tools import freeze_graph

#Global Variables
#================
MODEL_NAME = 'mnist_model_cnn'
SAVED_MODEL_PATH = 'mnist_cnn_saved_model/'
TRAIN_STEPS = 20000
BATCH_SIZE = 50
LEARNING_RATE = 1e-4

#Functions
#=========
#Function: conv2d
#Description: Returns a 2d convolution layer with full stride
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Function: max_pool_2x2
#Description: Returns a 2X downsampled feature map
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#Function: weight_variable
#Description: Returns a weight variable of the given shape
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#Function: bias_variable
#Description: Returns a bias variable of the given shape
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Main
#====
def main():
  #Read MNIST dataset
  #------------------
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  
  #Create The Model
  #----------------
  #Define input tensor
  #--Each entry of the tensor is a pixel intensity between 0 and 1
  #--x = [number of images, width*height of image]
  x = tf.placeholder(tf.float32, shape=[None, 784], name='modelInput')
  
  #Define loss and optimizer tensor
  #--This tensor holds the actual distribution for the labels of each image.
  #--y_ = [number of images, number of classes]
  y_ = tf.placeholder(tf.float32, [None, 10],"inputLabels")

  # Reshape input to use within a convolutional neural net.
  #--x_image = [batch_size, image_height, image_width, channels]
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  #CONVOLUTION 1
  # First convolutional layer - maps input image to 32 5x5 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  #POOLING 1
  # Pooling layer - downsamples h_conv1 by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  #CONVOLUTION 2
  # Second convolutional layer -- maps h_pool1 to 64 5x5 feature maps.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  #POOLING 2
  # Second pooling layer - downsamples h_conv2 by 2X
  h_pool2 = max_pool_2x2(h_conv2)

  #FULLY CONNECTED 1
  # Fully connected layer 1 -- after 2 round of downsampling, the input 28x28 image
  # is down to 7x7x64 feature maps -- map this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32, name='keepProb')
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  #FULLY CONNECTED 2
  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  #Define convolutional neural network output of model
  #--Softmax serves as an activation function that normalizes the evidence into 
  #  a probability distribution that an image is in some class
  y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='modelOutput')

  #Define Cross Entropy
  #---------------------------------- 
  #--Cross entropy is a measure of how inefficient our predictions are at describing 
  #  the actual labels of the images.  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  #Train the model by minimizing cross entropy
  #--By minizing cross entropy (the loss), the better the model is at predicting the correct
  #  label 
  train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

  #Train Model
  #-----------
  sess = tf.Session()
  init = tf.global_variables_initializer()

  sess.run(init)
  saver = tf.train.Saver()

  #Determine if the predicited class matches the actual label 
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  
  
  #Determine the percentage of images our model correctly predicted
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  #Run the training step N times
  for i in range(TRAIN_STEPS+1):
    print('Training Step:'+str(i))
  	#Run training step for batch of K images
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if i%100==0:
        #Print accuracy and loss
        print('  Accuracy = '+ str(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100)+"%" +
    	        '	 Loss = ' + str(sess.run(cross_entropy, {x: batch_xs, y_: batch_ys, keep_prob: 1.0}))
    	       )
    #Save learn weights of model to checkpoint file 
    if i%1000==0:
    	out = saver.save(sess, SAVED_MODEL_PATH+MODEL_NAME+'.ckpt', global_step=i)

  #Print final accuracy of model 
  print('Final Accuracy: ' + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))

  #Save model definition  
  tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pbtxt')
  tf.train.write_graph(sess.graph_def, SAVED_MODEL_PATH , MODEL_NAME + '.pb',as_text=False)

  #Freeze the graph
  #----------------
  #Input graph is our saved model defined above
  input_graph = SAVED_MODEL_PATH+MODEL_NAME+'.pb'
  #Use default graph saver
  input_saver = ""
  #Input file is a binary file
  input_binary = True
  #Checkpoint file to merge with graph definition
  input_checkpoint = SAVED_MODEL_PATH+MODEL_NAME+'.ckpt-'+str(TRAIN_STEPS)
  #Output nodes in model
  output_node_names = 'modelOutput'
  restore_op_name = 'save/restore_all'
  filename_tensor_name = 'save/Const:0'
  #Output path
  output_graph = SAVED_MODEL_PATH+'frozen_'+MODEL_NAME+'.pb'
  clear_devices = True
  initializer_nodes = ""
  #Freeze
  freeze_graph.freeze_graph(
    input_graph,
    input_saver,
    input_binary,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph,
    clear_devices,
    initializer_nodes,
  )

#RUN MAIN
#========
if __name__ == '__main__':
  main()