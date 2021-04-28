import pandas as pd
import numpy as np
import collections 
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from functools import partial

from   data_provider import load_data
from   data_provider import split_data
from   DropConnect import DropConnectModel
import UncertaintyM as unc
import tensorflow as tf
from   data_provider import load_data
from sklearn import preprocessing


# num_runs = 10
# Hyperprameters
epochs = 2
batch_size = 32
dropconnect_prob = 0.2
MC_samples = 50
# train_test_split = 0.2
# g_data_name = "cifar10"#"LDB_EURUSD" "mnist"
model_plot_flag = False


def run_BDL_model(x_train_, x_test_, Y_train, Y_test, seed):
	run_i = seed
	np.random.seed(run_i)
	# Loading and preprocessing data
	num_classes = len(np.unique(Y_train))

	normalizer = preprocessing.StandardScaler().fit(x_train_)
	x_train = normalizer.transform(x_train_)
	x_test = normalizer.transform(x_test_)

	y_train = tf.keras.utils.to_categorical(Y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(Y_test, num_classes)
	# print(y_test)
	# exit()

	# Model creation
	model = DropConnectModel(num_classes=num_classes, prob=dropconnect_prob, use_dropConnect=True)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	r = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
	# print(f"[run:{run_i+1}] test score:", model.evaluate(x_test, y_test, verbose=0))

	# MC sampling for Bayesian inference                                                                    TTTTTTTTTTTTTTTTTTTTTT needs time optimization
	MC_samples_predictions_prob = []
	for x in range(MC_samples):
		MC_samples_predictions_prob.append(model.predict(x_test))
		# print(f"MC_sample {x+1} out of {MC_samples} is done.")

	if model_plot_flag:
		plt.plot(r.history["loss"], label="loss")
		plt.plot(r.history["val_loss"], label="val_loss")
		plt.legend()
		plt.savefig(f"./pic/BDL_loss_run{run_i+1}.png")
		plt.close()

		plt.plot(r.history["accuracy"], label="acc")
		plt.plot(r.history["val_accuracy"], label="val_acc")
		plt.legend()
		plt.savefig(f"./pic/BDL_acc_run{run_i+1}.png")
		plt.close()


	# Uncertainty estimates
	predictions_temp = [[[] for j in range(MC_samples)] for i in range(x_test.shape[0])]
	porb_matrix = a = [[[] for j in range(MC_samples)] for i in range(x_test.shape[0])]


	for model_index, model_prediction in enumerate(MC_samples_predictions_prob):
		for data_index in range(x_test.shape[0]):
			porb_matrix[data_index][model_index] = model_prediction[data_index]
			predictions_temp[data_index][model_index] = np.argmax(model_prediction[data_index])


	prediction = []
	for prob_predic_data in predictions_temp:
		counter = collections.Counter(prob_predic_data)
		temp = collections.Counter(counter)
		prediction.append(temp.most_common()[0][0])

	total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, = unc.uncertainty_ent(np.array(porb_matrix))
	# labels = np.array(list(Y_test))
	return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty , model


# if __name__ == '__main__':

#     prediction_list = []
#     total_uncertainty_list = []
#     epistemic_uncertainty_list = []
#     aleatoric_uncertainty_list = []
#     labels = []
#     features, target = load_data(g_data_name)
#     func = partial(run_BDL_model,features, target, g_data_name ,train_test_split)
#     with Pool(12) as p:
#         res = p.map(func, range(num_runs))

#     for run in res:
#         prediction_list.append(run[0])
#         total_uncertainty_list.append(run[1])
#         epistemic_uncertainty_list.append(run[2])
#         aleatoric_uncertainty_list.append(run[3])
#         labels.append(run[4])

#     unc.accuracy_rejection(prediction_list,labels,total_uncertainty_list,    plot_name= f"BDL_total_"+g_data_name+ f"_run{num_runs}")
#     unc.accuracy_rejection(prediction_list,labels,epistemic_uncertainty_list,plot_name= f"BDL_epist_"+g_data_name+ f"_run{num_runs}")
#     unc.accuracy_rejection(prediction_list,labels,aleatoric_uncertainty_list,plot_name= f"BDL_ale_"  +g_data_name+ f"_run{num_runs}")
