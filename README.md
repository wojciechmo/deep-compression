## Deep compression

TensorFlow implementation of paper: Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.

The goal is to compress the neural network using weights pruning and weights quantization with no loss of accuracy.

Neural network architecture:

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/architecture.png" width="300">

Training accuracy chart:

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/train_accuracy.png" width="800">

### 1. Full trainig.

Train for number of iterations with gradient descent adjusting all the weights in every layer.

### 2. Pruning and finetuning.

Once in a while remove weights lower than a threshold. In the meantime finetune remaining weights to recover accuracy.

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/prune.png" width="650">

### 3. Quantization and finetuning.

Cluster remainig weights using k-means. Ater that finetune quantized weights for the remaining sparse connections to recover accuracy. Each layer weights are independently quantized.

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/quantize.png" width="650">

### 4. Deployment.

Fully connected layers are done as sparse matmul operation. TensorFlow doesn't allow to do sparse convolutions. Convolution layers are explicitly transformed to sparse matrix operations with full control over valid weights. 

Convolution as matrix operation:

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/sparse_conv1.png" width="500">

Convolution as matrix operation: 

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/sparse_conv2.png" width="250">







