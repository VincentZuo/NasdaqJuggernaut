from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

class TextPredModel:
	def __init__(self, encoder):
		self.encoder = encoder
		self.model = self.constructModel()

	# Virtual method used by child class.
	def constructModel(self, encoder):
		raiseNotDefined()

	def train(self, traning_data, validation_data, epochs=10, validation_steps=30):
		history = model.fit(train_dataset, epochs, validation_data, validation_steps)
		test_loss, test_acc = model.evaluate(test_dataset)
		return (history, test_loss, test_acc)

	def infer(self, infer_data):
		raiseNotDefined()

class SequentialModel(TextPredModel):
	def constructModel(self):
		model = tf.keras.Sequential([
	    tf.keras.layers.Embedding(self.encoder.vocab_size, 64),
	    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
	    tf.keras.layers.Dense(64, activation='relu'),
	    tf.keras.layers.Dense(1, activation='sigmoid')])
		model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
		return model

