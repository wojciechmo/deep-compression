import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

class LayerTrain(object):
	
	def __init__(self, in_depth, out_depth, N_clusters, name):
		
		self.name = name
		
		if 'conv' in name:
			self.w = tf.Variable(tf.random.normal([5, 5, in_depth, out_depth], stddev=0.1))

		elif 'fc' in name:
			self.w = tf.Variable(tf.random.normal([in_depth, out_depth], stddev=0.1))
			
		self.w_PH = tf.placeholder(tf.float32, self.w.shape) 
		self.assign_w = tf.assign(self.w, self.w_PH)
		self.num_total_weights = np.prod(self.w.shape)
		
		# mask placeholder for pruning
		# ones - valid weights, zero - pruned weights
		self.pruning_mask_data = np.ones(self.w.shape, dtype=np.float32)
		
		self.N_clusters = N_clusters; # for quantization
	
	def forward(self,x):
		
		if 'conv' in self.name:
			return tf.nn.conv2d(x, self.w, strides=[1, 2, 2, 1], padding='SAME')
			
		elif 'fc' in self.name:		
			return tf.matmul(x, self.w)
				
	def save_weights_histogram(self, sess, directory, iteration):
		
		w_data = sess.run(self.w).reshape(-1)	
		valid_w_data = [x for x in w_data if x!=0.0]
		
		plt.grid(True)
		plt.hist(valid_w_data, 100, color='0.4')
		plt.gca().set_xlim([-0.4, 0.4])
		plt.savefig(directory + '/' + self.name + '-' + str(iteration), dpi=100)
		plt.gcf().clear()

	def save_weights(self, sess, directory):
		
		w_data = sess.run(self.w)		
		np.save(directory + '/' + self.name + '-weights', w_data)
		np.save(directory + '/' + self.name + '-prune-mask', self.pruning_mask_data)
		
	# ------------------------------------------------------------------------		
	# ----------------------------- pruning ----------------------------------
	# ------------------------------------------------------------------------
		
	# even though it's enough to impact forward pass in each train step and leave gradients updates intact
	# here impact only backward pass in each step - to stay consistent with quantization
	# theoretically weights can be pruned only once and from that point on gradients should be pruned in every iteration
	# but for numerical stability weights are also pruned in every iteration 
	
	def prune_weights(self, sess, threshold):
		
		w_data = sess.run(self.w)
		# prune weights that are smaller then threshold - make them zero
		self.pruning_mask_data = (np.abs(w_data) >= threshold).astype(np.float32) 

		print 'layer:', self.name
		print '\tremaining weights:', int(np.sum(self.pruning_mask_data))	
		print '\ttotal weights:', self.num_total_weights	
						
		sess.run(self.assign_w, feed_dict={self.w_PH: self.pruning_mask_data*w_data})

	def prune_weights_gradient(self, grad):
		
		return grad * self.pruning_mask_data
	
	# for numerical stability
	def prune_weights_update(self, sess):
		
		w_data = sess.run(self.w)
		sess.run(self.assign_w, feed_dict={self.w_PH: self.pruning_mask_data*w_data})
			
	# ------------------------------------------------------------------------
	# --------------------------- quantization -------------------------------
	# ------------------------------------------------------------------------
		
	def quantize_weights(self, sess):

		w_data = sess.run(self.w)
		
		# theoretically pruning mask should be taken into consideration to compute max and min data only among valid weights
		# but in practice with normal ditribution init there is 100% chances that min and max vals will be among valid weights
		max_val = np.max(w_data)
		min_val = np.min(w_data)

		# linearly initialize centroids between max and min
		self.centroids = np.linspace(min_val, max_val, self.N_clusters)
		w_data = np.expand_dims(w_data, 0)

		centroids_prev = np.copy(self.centroids)

		for i in range(20):
			
			if 'conv' in self.name:
				distances = np.abs(w_data - np.reshape(self.centroids,(-1, 1, 1, 1, 1)))
				distances = np.transpose(distances, (1,2,3,4,0))
				
			elif 'fc' in self.name:
				distances = np.abs(w_data - np.reshape(self.centroids,(-1, 1, 1)))
				distances = np.transpose(distances, (1,2,0))				
				
			classes = np.argmin(distances, axis=-1)

			self.clusters_masks = []
			for i in range(self.N_clusters):
				
				cluster_mask = (classes == i).astype(np.float32) * self.pruning_mask_data
				self.clusters_masks.append(cluster_mask) 
			
				num_weights_assigned = np.sum(cluster_mask)
				
				if num_weights_assigned!=0:
					self.centroids[i] = np.sum(cluster_mask * w_data) / num_weights_assigned
				else: # do not modify
					pass
			
			if np.array_equal(centroids_prev, self.centroids):
				break
				
			centroids_prev = np.copy(self.centroids)
		
		self.quantize_weights_update(sess)

		print 'layer:', self.name
		print '\tcentroids:', self.centroids
		
	def group_and_reduce_weights_gradient(self, grad):
		
		grad_out = np.zeros(self.w.shape, dtype=np.float32)
		
		for i in range(self.N_clusters):
			
			cluster_mask = self.clusters_masks[i]
			centroid_grad = np.sum(grad * cluster_mask)
			
			grad_out = grad_out + cluster_mask * centroid_grad
			
		return grad_out
		
	# for numerical stability	
	def quantize_centroids_update(self, sess):

		w_data = sess.run(self.w)

		for i in range(self.N_clusters):
			
			cluster_mask = self.clusters_masks[i]
			cluster_count = np.sum(cluster_mask)
			
			if cluster_count!=0:
				self.centroids[i] = np.sum(cluster_mask * w_data) / cluster_count
			else: # do not modify
				pass
				
	# for numerical stability
	def quantize_weights_update(self, sess):
		
		w_data_updated = np.zeros(self.w.shape, dtype=np.float32)
		
		for i in range(self.N_clusters):
			
			cluster_mask = self.clusters_masks[i]
			centroid = self.centroids[i]
			
			w_data_updated = w_data_updated + cluster_mask * centroid

		sess.run(self.assign_w, feed_dict={self.w_PH: self.pruning_mask_data * w_data_updated})

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

class FcLayerDeploy(object):
	
	def __init__(self, matrix, prune_mask, name, dense=False):
			
		assert matrix.shape == prune_mask.shape
		
		self.dense = dense
		self.N_in, self.N_out = matrix.shape
		
		if self.dense == False:
			
			indices, values, dense_shape = [], [], [self.N_in, self.N_out] # sparse matrix representation
			
			for i in range(self.N_in):
				for j in range(self.N_out):
					
					# pruning mask: ones - valid weights, zero - pruned weights
					if prune_mask[i][j] == 0.0:
						continue
					
					indices.append([i,j])
					values.append(matrix[i][j])
			
			self.w_matrix = tf.SparseTensor(indices, values, dense_shape) # tf sparse matrix	
			
		else:
			
			self.w_matrix = tf.constant(matrix * prune_mask) # tf dense matrix
		
		print 'layer:', name
		print '\tvalid matrix weights:', np.sum(prune_mask)
		print '\ttotal matrix weights:', np.product(self.w_matrix.shape)
				
	def forward_matmul(self, x):
		
		if self.dense == False:
			
			w = tf.sparse.transpose(self.w_matrix, (1, 0))
			x = tf.transpose(x, (1, 0))
			x = tf.sparse.matmul(w, x) # only left matrix can be sparse hence transpositions
			x = tf.transpose(x ,(1, 0))	
				
		else:

			x = tf.matmul(x, self.w_matrix)	
		
		return x

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

class ConvLayerDeploy(object):
	
	def __init__(self, tensor, prune_mask, H_in, W_in, stride, name, dense=False):
		
		assert tensor.shape == prune_mask.shape
		
		self.stride = stride
		self.dense = dense

		if self.dense == False:
			indices, values, dense_shape = self.tensor_to_matrix(tensor, prune_mask, H_in, W_in, stride)
			self.w_matrix = tf.SparseTensor(indices, values, dense_shape) # tf sparse matrix
			
		else:
			matrix = self.tensor_to_matrix(tensor, prune_mask, H_in, W_in, stride)
			self.w_matrix = tf.constant(matrix) # tf dense matrix	
		
		self.w_tensor = tf.constant(tensor * prune_mask)

		print 'layer:', name
		print '\tvalid matrix weights:', int(np.sum(prune_mask))
		print '\ttotal tensor weights:', np.product(self.w_tensor.shape)
		print '\ttotal matrix weights:', np.product(self.w_matrix.shape)

	def get_linear_pos(self, i, j, W):	# row major
		
		return i * W + j
			
	def tensor_to_matrix(self, tensor, prune_mask, H_in, W_in, stride):
				
		# assume padding type 'SAME' and padding value 0 

		H_out = int(H_in +1)/stride # padding 'SAME'
		W_out = int(W_in +1)/stride # padding 'SAME'
		H_in = int(H_in)
		W_in = int(W_in)
					
		kH, kW, D_in, D_out = tensor.shape 
		
		self.D_out = D_out
		self.H_out = H_out
		self.W_out = W_out
 
		if self.dense == False:
			indices, values, dense_shape = [], [], [H_in * W_in * D_in, H_out* W_out * D_out] # sparse matrix
		else:
			matrix = np.zeros((H_in * W_in * D_in, H_out* W_out * D_out), dtype=np.float32) # dense matrix		
		
		for d_in in range(D_in):
			for d_out in range(D_out):
		
				# tf.nn.conv2d implementation doesn't go from top-left spatial location but from bottom-right
				for i_in_center in np.arange(H_in-1, -1, -stride): # kernel input center for first axis
					for j_in_center in np.arange(W_in-1, -1, -stride): # kernel input center for second axis
								
						i_out = int(i_in_center / stride)
						j_out = int(j_in_center / stride)

						for i in range(kH):
							
							i_in = i_in_center + i - kH/2
								
							if i_in < 0 or i_in >= H_in: # padding value 0 
								continue 
													
							for j in range(kW):
								
								j_in = j_in_center + j -kW/2
								
								if j_in < 0 or j_in >= W_in: # padding value 0 
									continue
								
								# pruning mask: ones - valid weights, zero - pruned weights
								if prune_mask[i][j][d_in][d_out] == 0.0:
									continue		
								
								pos_in = self.get_linear_pos(i_in, j_in, W_in) + d_in * H_in * W_in
								pos_out = self.get_linear_pos(i_out, j_out, W_out) + d_out * H_out * W_out
								
								if self.dense == False:
									indices.append([pos_in, pos_out])
									values.append(tensor[i][j][d_in][d_out])
								else:
									matrix[pos_in][pos_out] = tensor[i][j][d_in][d_out]

		if self.dense == False:
			return indices, values, dense_shape
		else:
			return matrix

	def forward_matmul_preprocess(self, x):
		
		x = tf.transpose(x, (0, 3, 1, 2))
		x = tf.reshape(x,(-1, np.product(x.shape[1:])))
		
		return x
		
	def forward_matmul_postprocess(self, x):

		x = tf.reshape(x,(-1, self.D_out, self.H_out, self.W_out))
		x = tf.transpose(x, (0, 2, 3, 1))
	
		return x
		
	def forward_matmul(self, x):
		
		if self.dense == False:
			w = tf.sparse.transpose(self.w_matrix, (1, 0))
			x = tf.transpose(x, (1, 0))
			x = tf.sparse.matmul(w, x) # only left matrix can be sparse hence transpositions
			x = tf.transpose(x, (1, 0))
		else:
			x = tf.matmul(x, self.w_matrix)
		
		return x
		
	def forward_conv(self, x):
		
		return tf.nn.conv2d(x, self.w_tensor, strides=[1, self.stride, self.stride, 1], padding='SAME')
