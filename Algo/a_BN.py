import ray
import pandas as pd
import numpy as np
import collections 
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from functools import partial

# from   DropConnect import DropConnectModel
import UncertaintyM as unc
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
batch_size = 16


def BN_init(x_train, x_test, y_train, y_test, pram, unc_method, seed):
	run_i = seed
	np.random.seed(run_i)
	tf.random.set_seed(seed)
	# Loading and preprocessing data
	num_classes = len(np.unique(y_train))

	y_train_N = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test_N = tf.keras.utils.to_categorical(y_test, num_classes)

	# Model creation
	# model = DropConnectModel(num_classes=num_classes, prob=pram["dropconnect_prob"], use_dropConnect=False)  # DropConnect model

	# model = tf.keras.models.Sequential([ # normal dropout model
	# 	tf.keras.layers.Dense(64, activation='relu'),
	# 	tf.keras.layers.Dropout(pram["dropconnect_prob"],seed=seed),
	# 	tf.keras.layers.Dense(64, activation='relu'),
	# 	tf.keras.layers.Dropout(pram["dropconnect_prob"],seed=seed),
	# 	tf.keras.layers.Dense(num_classes, activation='softmax')
	# 	])
	# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /  tf.cast(len(x_train), dtype=tf.float32))# pylint: disable=g-long-lambda

	model = tf.keras.models.Sequential([
			tfp.layers.DenseFlipout(5, kernel_divergence_fn=kl_divergence_function,activation=tf.nn.relu),
			tfp.layers.DenseFlipout(num_classes, kernel_divergence_fn=kl_divergence_function,activation=tf.nn.softmax)
			# tf.keras.layers.Dense(5, activation='relu'),
			# tf.keras.layers.Dense(num_classes, activation='softmax'),
			])
	model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

	return y_train_N, y_test_N, model


def BN_run(x_train, x_pool, y_train, y_pool, x_test, y_test, pram, unc_method, seed, model,active_step, log=False):
	us = unc_method.split('_')
	unc_method = us[0]
	if len(us) > 1:
		unc_mode = us[1] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that
	run_i = seed
	np.random.seed(run_i)
	# Model creation
	training_epochs = pram["epochs"]
	if active_step==0:
		training_epochs = pram["init_epochs"]

	r = model.fit(x_train, y_train, epochs=training_epochs, batch_size=batch_size, verbose=0) #  validation_data=(x_pool_N, y_pool_N)
	
	if "ent" == unc_method:
		all_data = np.concatenate((x_pool,x_test),axis=0) # combine all train pool and test data 
		MC_prob_all   = mc_sampling(model, all_data, pram["MC_samples"]) # sample all data at once
		#seperating the data
		MC_prob_pool  = MC_prob_all[:,0:len(x_pool)]
		MC_prob_test  = MC_prob_all[:,len(x_pool):]

		# Uncertainty estimates
		porb_matrix = a = [[[] for j in range(pram["MC_samples"])] for i in range(x_pool.shape[0])]

		for model_index, model_prediction in enumerate(MC_prob_pool):
			for data_index in range(x_pool.shape[0]):
				porb_matrix[data_index][model_index] = model_prediction[data_index]

		total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent(np.array(porb_matrix))

	elif "sent" == unc_method:
		porb_matrix = model.predict(x_pool,verbose=0)
		total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_standard(porb_matrix)

	elif "rl" == unc_method:
		all_data = np.concatenate((x_train,x_pool,x_test),axis=0) # combine all train pool and test data 
		MC_prob_all   = mc_sampling(model, all_data, pram["MC_samples"]) # sample all data at once
		#seperating the data
		MC_prob_train = MC_prob_all[:,0:len(x_train)]
		MC_prob_pool  = MC_prob_all[:,len(x_train):len(x_train)+len(x_pool)]
		MC_prob_test  = MC_prob_all[:,len(x_train)+len(x_pool):]

		# temp_log = False
		# if active_step == 2 or active_step == 14:
		# 	temp_log = True
		
		total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.unc_rl_prob(MC_prob_train, MC_prob_pool, y_train) # calculating unc
	elif "random" == unc_method:
		MC_prob_test   = mc_sampling(model, x_test, pram["MC_samples"]) # sample all data at once

		total_uncertainty = np.random.rand(len(x_pool))
		epistemic_uncertainty = np.random.rand(len(x_pool))
		aleatoric_uncertainty = np.random.rand(len(x_pool))
	else:
		print(f"[Error] No implementation of unc_method {unc_method} for BN")

	# calculating acc on the test data
	prob_test = np.mean(MC_prob_test,axis=0)
	y_pred = np.argmax(prob_test, axis=1)
	acc = accuracy_score(y_test, y_pred)
	
	if log:
		print(">>> debug model.summary \n",model.summary())


	return acc, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty , model


# @ray.remote
# def mc_sampling_ray(model, x_data, sample_count):
# 	return model.predict(x_data,verbose=0)

# def mc_sampling(model, x_data, sample_count, paralel=True):
# 	if paralel:
# 		ray_array = []
# 		for x in range(sample_count):
# 			ray_array.append(mc_sampling_ray.remote(model, x_data, sample_count))
# 		res_array = ray.get(ray_array)
# 		return res_array
# 	else:	
# 		MC_samples_predictions_prob = []
# 		for x in range(sample_count):
# 			MC_samples_predictions_prob.append(model.predict(x_data,verbose=0))
# 		return np.array(MC_samples_predictions_prob)


def mc_sampling(model, x_data, sample_count):
	MC_samples_predictions_prob = []
	for x in range(sample_count):
		# MC_samples_predictions_prob.append(model.predict(x_data,verbose=0))
		MC_samples_predictions_prob.append(model(x_data,training=True))
		weights = model.get_weights()
		print("------------------------------------")
		for lay in model.layers:
			print(lay.name)
			print(lay.get_weights())
		print("------------------------------------")
		print(weights)
		print(sd)
	return np.array(MC_samples_predictions_prob)