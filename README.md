# MNIST-Classifier-App
Android app to classify hand drawn numbers using Tensorflow machine learning model. 

## Creating the model
Before creating the App, we must first create a model in tensorflow. The model is created with Python and is then
frozen and imported inside Android Studio. The app supports two different classification models; a single layer SoftMax model
and a Convolutional Neural Network. The SoftMax model has an final accurary of ~92% while the Convolutional Neural Network increases
the accuracy to ~98%.

### SoftMax 
The SoftMax model code can be found at [mnist_model.py](../model/mnist_model.py).

#### Input
The input to the model is the MNIST data set, which contains 28x28 pixel images of handwritten numbers.
Each image also has a 1 hot vector that defines the class associated with it. 
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
allows the activation function to shift to the left or right.
```python
b = tf.Variable(tf.zeros([10]), name='modelBias')
```
#### Output
The output of the model is given by a layer of SoftMax that acts as an activation function that normalizes
the evidence into a probability distribution that an image is in some class. Evidence is given by the
function:
*'evidence=sum(x*W + b)'*

```python
y = tf.nn.softmax(tf.matmul(x, W) + b, name='modelOutput')
```

### Convolutional Neural Network
