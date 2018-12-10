import tensorflow as tf
import numpy as np

from layers import FcLayerDeploy, ConvLayerDeploy

def test_ConvLayerDeploy():
	
	weights = np.random.uniform(-10.0, 10.0, size=(3, 3, 4, 5)).astype(np.float32)
	prune_mask = np.random.uniform(0.0, 2.0, size=(3, 3, 4, 5)).astype(np.int32).astype(np.float32)

	x = np.random.uniform(0.0, 1.0, size=(2, 14, 14, 4)).astype(np.float32)
	x = tf.constant(x)

	L = ConvLayerDeploy(weights, prune_mask, 14, 14, 2, 'conv')

	x_matmul_sparse = L.forward_matmul_preprocess(x)
	y_matmul_sparse = L.forward_matmul(x_matmul_sparse)
	y_matmul_sparse = L.forward_matmul_postprocess(y_matmul_sparse)

	y_conv = L.forward_conv(x)

	sess = tf.Session()
	y_matmul_sparse_data, y_conv_data = sess.run([y_matmul_sparse, y_conv])

	assert np.mean(np.abs(y_matmul_sparse_data - y_conv_data)) < 1e-6

def test_FcLayerDeploy():
	
	weights = np.random.uniform(-10.0, 10.0, size=(5, 10)).astype(np.float32)
	prune_mask = np.random.uniform(0.0, 2.0, size=(5, 10)).astype(np.int32).astype(np.float32)
	
	x = np.random.uniform(0.0, 1.0, size=(2, 5)).astype(np.float32)
	x = tf.constant(x)
	
	L_sparse = FcLayerDeploy(weights, prune_mask, 'fc')
	L_dense = FcLayerDeploy(weights, prune_mask, 'fc', dense=True)

	y_sparse = L_sparse.forward_matmul(x)
	y_dense = L_dense.forward_matmul(x)
		
	sess = tf.Session()
	y_sparse_data, y_dense_data = sess.run([y_sparse, y_dense])

	assert np.mean(np.abs(y_sparse_data - y_dense_data)) < 1e-6
