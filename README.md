# Deep compression

TensorFlow implementation of paper: Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.



Training accuracy chart:
<img src="https://github.com/WojciechMormul/deep-compression/tree/master/imgs/train_accuracy.png" width="200">

1. Full trainig.

2. Pruning and finetuning.

3. Quantization and finetuning.
Each layer weights are independently quantized.


4. Deployment.

TensorFlow doesn't allow to do sparse convolutions. I transform convolution operation to sparse matmul with full control over valid weights.
