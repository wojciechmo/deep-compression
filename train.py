import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil

from layers import LayerTrain

def make_dir(directory):

	if os.path.exists(directory):
		shutil.rmtree(directory, ignore_errors=True)
	os.makedirs(directory)
	
if __name__ == "__main__":
	
	histograms_dir = './histograms'
	weights_dir = './weights'
	
	make_dir(histograms_dir)
	make_dir(weights_dir)
	
	L1 = LayerTrain(1, 32, N_clusters=5, name='conv1')
	L2 = LayerTrain(32, 64, N_clusters=5, name='conv2')
	L3 = LayerTrain(7 * 7 * 64, 1024, N_clusters=5, name='fc1')
	L4 = LayerTrain(1024, 10, N_clusters=5, name='fc2')

	LAYERS = [L1, L2, L3, L4]
	LAYERS_WIEGHTS = [L1.w, L2.w, L3.w, L4.w]

	x_PH = tf.placeholder(tf.float32, [None, 28, 28, 1])
	x = tf.nn.relu(L1.forward(x_PH))
	x = tf.nn.relu(L2.forward(x))
	x = tf.reshape(x, (-1, int(np.product(x.shape[1:])))) 
	x = tf.nn.relu(L3.forward(x))
	logits = L4.forward(x)

	preds = tf.nn.softmax(logits)
	labels = tf.placeholder(tf.float32, [None, 10])
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

	optimizer = tf.train.AdamOptimizer(1e-4)
	gradients_vars = optimizer.compute_gradients(loss, LAYERS_WIEGHTS)
	grads = [grad for grad, var in gradients_vars]
	train_step = optimizer.apply_gradients(gradients_vars)

	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	iters = []
	iters_acc = []
	
	for i in range(1500):
				
		batch_x, batch_y = mnist.train.next_batch(50)
		batch_x = np.reshape(batch_x,(-1, 28, 28,1))
		
		feed_dict={x_PH: batch_x, labels: batch_y}
		
		# ------------------------------------------------------		
		# --------------- full network training ----------------
		# ------------------------------------------------------
		
		if i < 500:
			sess.run(train_step, feed_dict=feed_dict)
		
		# ------------------------------------------------------		
		# ---------------------- pruning -----------------------
		# ------------------------------------------------------
		
		elif i >= 500 and i < 1000: 
			
			# prune from time to time, finetune in the meantime
			if i%500==0:
				print 'iter:', i, 'prune weights'
				for L in LAYERS:
					L.prune_weights(sess, threshold=0.1)
			
			grads_data = sess.run(grads, feed_dict={x_PH: batch_x, labels: batch_y})
			feed_dict = {}
			for L, grad, grad_data in zip(LAYERS, grads, grads_data):
				pruned_grad_data = L.prune_weights_gradient(grad_data)
				feed_dict[grad] = pruned_grad_data
				
			sess.run(train_step, feed_dict=feed_dict)
			
			# for numerical stability
			for L in LAYERS:
				L.prune_weights_update(sess)
		
		# ------------------------------------------------------		
		# ------------------- quantization ---------------------
		# ------------------------------------------------------

		else:
			
			# quantize only once and then finetune
			if i==1000:
				print 'iter:', i, "quantize weights"
				for L in LAYERS:
					L.quantize_weights(sess)

			grads_data = sess.run(grads, feed_dict={x_PH: batch_x, labels: batch_y})
			feed_dict = {}
			for L, grad, grad_data in zip(LAYERS, grads, grads_data):
				grouped_grad_data = L.group_and_reduce_weights_gradient(grad_data)
				feed_dict[grad] = grouped_grad_data
			
			sess.run(train_step, feed_dict=feed_dict)
			
			# for numerical stability		
			for L in LAYERS:
				L.quantize_centroids_update(sess)
				L.quantize_weights_update(sess)
				
		# ------------------------------------------------------		
		# --------------------- evaluation ---------------------
		# ------------------------------------------------------
		
		if i%10 == 0:
			
			batches_acc = []
			for j in range(10):
			
				batch_x, batch_y = mnist.test.next_batch(1000)
				batch_x = np.reshape(batch_x,(-1, 28, 28,1))	
			
				batch_acc = sess.run(accuracy,feed_dict={x_PH: batch_x, labels: batch_y})
				batches_acc.append(batch_acc)
						
			acc = np.mean(batches_acc)

			iters.append(i)
			iters_acc.append(acc)				
			print 'iter:', i, 'test accuracy:', acc

			for L in LAYERS:
				L.save_weights_histogram(sess, histograms_dir, i)

	for L in LAYERS:
		L.save_weights(sess, weights_dir)

	plt.figure(figsize=(10, 4))
	plt.ylabel('accuracy', fontsize=12)
	plt.xlabel('iteration', fontsize=12)
	plt.grid(True)
	plt.plot(iters, iters_acc, color='0.4')
	plt.savefig('./train_acc', dpi=1200)

	print 'Training finished'
