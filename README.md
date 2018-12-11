## Deep compression

TensorFlow implementation of paper: Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.

The goal is to compress the neural network using weights pruning and quantization with no loss of accuracy.

Neural network architecture:</br>
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/architecture.png" width="280">

Test accuracy during training:</br>
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/train_accuracy.png" width="700">

### 1. Full trainig.

Train for number of iterations with gradient descent adjusting all the weights in every layer.

### 2. Pruning and finetuning.

Once in a while remove weights lower than a threshold. In the meantime finetune remaining weights to recover accuracy.</br>
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/prune.png" width="650">

### 3. Quantization and finetuning.

Quantization is done after pruning. Cluster remainig weights using k-means. Ater that finetune centroids of remaining quantized weights to recover accuracy. Each layer weights are quantized independently.</br>
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/quantize.png" width="650">

### 4. Deployment.

Fully connected layers are done as sparse matmul operation. TensorFlow doesn't allow to do sparse convolutions. Convolution layers are explicitly transformed to sparse matrix operations with full control over valid weights. 

Simple (input_depth=1, output_depth=1) convolution as matrix operation:</br>
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/sparse_conv1.png" width="500">

Full (input_depth>1, output_depth>1) convolution as matrix operation:</br>
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/sparse_conv2.png" width="260">

I do not make efficient use of quantization during deployment. It is possible to do it using TensorFlow operations, but it would be super slow, as for each output unit we need to create N_clusters sparse tensors from input data, reduce_sum in each tensor, multiply it by clusters and add tensor values resulting in output unit value. To do it efficiently, it requires to write kernel on GPU, which I intend to do in the future.
