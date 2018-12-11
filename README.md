# Deep compression

TensorFlow implementation of paper: Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.



Training accuracy chart:
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/train_accuracy.png" width="800">

1. Full trainig.

2. Pruning and finetuning.
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/prune.png" width="500">

3. Quantization and finetuning.
Each layer weights are independently quantized.
<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/quantize.png" width="500">


4. Deployment.

TensorFlow doesn't allow to do sparse convolutions. I transform convolution operation to sparse matmul with full control over valid weights.

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/sparse_conv1.png" width="500">

<img src="https://github.com/WojciechMormul/deep-compression/blob/master/imgs/sparse_conv2.png" width="250">



