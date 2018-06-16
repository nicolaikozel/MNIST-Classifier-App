# MNIST-Classifier-App
Android app to classify hand drawn numbers using Tensorflow machine learning model. 

## Creating the Model
Before creating the App, we must first create a model in tensorflow. The model is created with Python and is then
frozen and imported inside Android Studio. The app supports two different classification models; a single layer SoftMax model
and a Convolutional Neural Network. The SoftMax model has an final accurary of ~92% while the Convolutional Neural Network increases
the accuracy to ~98%.
### SoftMax 
The SoftMax model code can be found at [mnist_model.py](../model/mnist_model.py).
#### Input Layer
The input to the model is the MNIST data set, which contains 28x28 pixel images of handwritten numbers.
Each image also has a 1 hot vector that defines the class associated with it. 

![alt text](https://www.tensorflow.org/images/mnist_0-9.png)

```python
#--Each entry of the tensor is a pixel intensity between 0 and 1
#--x = [number of images, width*height of image]
x = tf.placeholder(tf.float32, shape=[None, 784], name='modelInput')
#--This tensor holds the actual distribution for the labels of each image.
#--y_ = [number of images, number of classes]
y_ = tf.placeholder(tf.float32, [None, 10],"inputLabels")
```
#### Weights
The weights of the model define how pixel intensity indicates a certain class.  For a pixel of high 
intensity, the weight is positive if it is evidence in favor of the image being in some class and negative if it is not. 
```python
#--W = [width*height of image, number of classes] 
W = tf.Variable(tf.zeros([784,10]), name='modelWeights')
```
#### Bias
The bias represents extra evidence that some things are more likely independent of the input. Mathematically, bias
allows the activation function to shift to the left or right. A higher bias allows for quicker training but the model is less flexible.
```python
#--b = [number of classes]
b = tf.Variable(tf.zeros([10]), name='modelBias')
```
#### Output Layer
The output of the model is given by a layer of SoftMax. This layer acts as an activation function that normalizes
the evidence into a probability distribution that an image is in some class. Evidence is given by the
function:
**evidence=x*W + b**
```python
#--y = [number of images, number of classes]
y = tf.nn.softmax(tf.matmul(x, W) + b, name='modelOutput')
```
### Convolutional Neural Network
The Convolutional Neural Network model code can be found at [mnist_model_cnn.py](../model/mnist_model_cnn.py).
#### Input Layer
The input of the model is the same as the SoftMax model, except we must reshape the input to use within a convolutional neural net.
This is because the tensorflow methods for creating convolutional and pooling layers expect input tensors of shape 
[batch_size, image_height, image_width, channels]. 
```python
#--x_image = [batch_size, image_height, image_width, channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])
```
#### Convolution Layer 1
Convolution layers are used to extract features from the input image. They work by convolving a filter around the input image and taking the dot product between them. This produces a feature map.
![alt text](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=268&h=196&zoom=2)

For this layer, we also apply ReLU as the activation function. 
**f(x)=max(0,x)**

In this first layer, we are mapping the input image to 32 different 5x5 feature maps.
```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
```
#### Pooling Layer 1
```python
h_pool1 = max_pool_2x2(h_conv1)
```
#### Convolution Layer 2
```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
```
#### Pooling Layer 2
```python
h_pool2 = max_pool_2x2(h_conv2)
```
## Training the Model
