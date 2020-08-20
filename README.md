# Deep Neuron Network Implementation without Toolkit

## Dataset
- MNIST

## Structure:
- **10 epochs**, **100 training data each mini-batch**
- Fully connected<br><br>
- **Input layer**
  - 784 neurons (28*28 pixels)
- **1st hidden layer**
  - 500 neurons
  - Activation function: ***relu*** (prevent gradient vanishing)
- **2nd hidden layer**
  - 500 neurons
  - Activation function: ***sigmoid*** (no special meaning)
- **Output layer**
  -  10 neurons
  - Activation function: ***softmax***
- **Loss function**
  - Cross Entropy

***structure.py*** and ***training.py*** should be in the same directory. Run ***training.py*** to start training. Loss value will be plotted for each batch and accuracy will be calculated automatically after the training phase. The accuracy I got is approximately 80%. If you wish to adjust the layer structure, modify the DNN class in ***structure.py***.

Currently the optimizer is ***SGD***, There are also ***Adam*** and ***RMSprop*** inside the code but I haven't tested it yet.

The [code](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html) for image preprocessing is from professor Hung-Yi Lee, which transforms each 28*28 image into an array with 784 length.
