### An interquantile multilayer perceptron
### The idea is to compute two selected nonlinear regression quantiles by means of multilayer perceptrons (MLPs)
### and the final regression fit is obtained as a standard MLP computed however only for such observations,
### which are between the two quantiles; to achieve robustness, the remaining observations are ignored completely.


### Importing libraries, preparing the prerequisities
import numpy as np
import math
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import load_model

### Interquantile MLP (the quantiles are determined by two constants denoted as tau1, tau2)
def IQ_MLP(input, output, tau1 = 0.15, tau2 = 0.85, batch_size = 20, qepochs = 500, epochs = 500, model = None, tens = False, save = True):

	def quantilLoss(quantile):
		def resLoss(y_true, y_pred):
			rez = (y_pred - y_true)
			boolTenzorGrather = K.cast(K.greater_equal(rez, K.zeros(K.shape(rez))), 'float32')
			boolTenzorLess = K.ones(K.shape(rez)) - boolTenzorGrather
			out = boolTenzorGrather*rez*quantile + boolTenzorLess*rez*(quantile-1)
			return K.mean(out, axis=-1)
		return resLoss
	
	def getModel():
		m = Sequential() 
		m.add(Dense(20, input_dim=input.shape[1])) ### Description of the (fixed) architecture
		m.add(LeakyReLU(alpha=.001))
		m.add(Dense(40))
		m.add(LeakyReLU(alpha=.001))
		m.add(Dense(40))
		m.add(LeakyReLU(alpha=.001))
		m.add(Dense(40))
		m.add(LeakyReLU(alpha=.001))
		m.add(Dense(1))
		return m
		
	net1Tau = getModel() ### Important place. Right here the nonlinear regression quantile by means of an MLP is trained.
	net1Tau.compile(loss=quantilLoss(tau1), optimizer='adam', metrics=['accuracy'])
	net1Tau.fit(input, output, batch_size=batch_size, epochs=qepochs, verbose=0, shuffle=True)
	
	net2Tau = getModel()
	net2Tau.compile(loss=quantilLoss(tau2), optimizer='adam', metrics=['accuracy'])
	net2Tau.fit(input, output, batch_size=batch_size, epochs=qepochs, verbose=0, shuffle=True)
	
	y1 = net1Tau.predict(input)
	y2 = net2Tau.predict(input)
	logLess = np.logical_and(np.less(output, np.reshape(y1, -1)), np.greater(output, np.reshape(y2, -1)))
	newInput = input[logLess]
	newOutput = output[logLess]
	### Now the idea of the interquantile estimator: only observations between the two quantiles are considered.
	if model == None:
		model = getModel()
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	
	if tens:
		NAME = "IQnet"
		tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))		
		model.fit(newInput, newOutput, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True, callbacks=[tensorboard])
	else:
		model.fit(newInput, newOutput, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True)

	if save:
		model.save('IQmodel.h5')
	return model

