#===MNIST Model===
#Author: Nicolai Kozel
#Description: Simple 1 layer softmax tensorflow model 
#to classify MNIST images. 

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
MODEL_NAME = 'mnist_model'
SAVED_MODEL_PATH = 'mnist_saved_model/'
TRAIN_STEPS = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.5

#Main
#====
def main():
  #Read MNIST dataset
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

  #Define weights tensor
  #--This tensor holds the information about how pixel intensity indicates a certain class. 
  #  For a pixel of high intensity, the weight is positive if it is evidence in favor of the
  #  image being in some class and negative if it is not. 
  #--W = [width*height of image, number of classes] 
  W = tf.Variable(tf.zeros([784,10]), name='modelWeights')

  #Define bias tensor
  #--This tensor holds the bias information; extra evidence to represent 
  #  that some things are more likely independent of the input.
  #--b = [number of classes]
  b = tf.Variable(tf.zeros([10]), name='modelBias')

  #Define softmax output of model
  #--Softmax serves as an activation function that normalizes the evidence into 
  #  a probability distribution that an image is in some class
  #--y = [number of images, number of classes]
  y = tf.nn.softmax(tf.matmul(x, W) + b, name='modelOutput')

  #Define Cross Entropy
  #---------------------------------- 
  #--Cross entropy is a measure of how inefficient our predictions are at describing 
  #  the actual labels of the images.  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  #Train the model by minimizing cross entropy
  #--By minizing cross entropy (the loss), the better the model is at predicting the correct
  #  label 
  train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

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

  #Run the training step 1000 times
  for i in range(TRAIN_STEPS+1):
  	#Run training step for batch of 100 images
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #Print accuracy and loss
    print('Training Step:' + str(i) +
    	  '  Accuracy = '+ str(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})*100)+"%" +
    	  '	 Loss = ' + str(sess.run(cross_entropy, {x: batch_xs, y_: batch_ys}))
    	 )
    #Save learned weights of model to checkpoint file 
    if i%5==0:
    	out = saver.save(sess, SAVED_MODEL_PATH+MODEL_NAME+'.ckpt', global_step=i)

  #Print final accuracy of model 
  print('Final Accuracy: ' + str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

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