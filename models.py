from __future__ import absolute_import, division, print_function, unicode_literals
from feature_engineering import pad_to_size

import tensorflow as tf

class TextPredModel:
	def __init__(self, encoder, emd_dim):
		self.encoder = encoder
		self.model = self.constructModel()
		self.emd_dim = emd_dim

	# Virtual method used by child class.
	def constructModel(self):
		raiseNotDefined()

	def train(self, traning_data, validation_data, epochs=10, validation_steps=30):
		history = self.model.fit(
			traning_data,
			epochs=epochs,
			validation_data=validation_data,
			validation_steps=validation_steps
		)
		test_loss, test_acc = self.model.evaluate(test_dataset)
		return (history, test_loss, test_acc)

	def infer(self, infer_string):
		encoded_infer_string = pad_to_size(encoder.encode(infer_string), self.emd_dim)
		encoded_infer_string = tf.cast(encoded_infer_string, tf.float32)
		return self.model.predict(tf.expand_dims(encoded_infer_string, 0))

class LstmModel(TextPredModel):
	def constructModel(self):
		model = tf.keras.Sequential([
			tf.keras.layers.Embedding(self.encoder.vocab_size, self.emd_dim),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.emd_dim)),
			tf.keras.layers.Dense(self.emd_dim, activation='relu'),
			# Always use 1 dimensional output
			tf.keras.layers.Dense(1, activation='sigmoid'
		)])
		model.compile(loss='binary_crossentropy',
			optimizer=tf.keras.optimizers.Adam(1e-4),
			metrics=['accuracy']
		)
		return model

class CnnModel(TextPredModel):
	def constructModel(self):
		model = tf.keras.Sequential([
			tf.keras.layers.Embedding(self.encoder.vocab_size, 64),
			tf.keras.layers.GlobalAveragePooling1D(),
			tf.keras.layers.Dense(1, activation='sigmoid')
		])
		model.compile(loss='binary_crossentropy',
			optimizer=tf.keras.optimizers.Adam(1e-4),
			metrics=['accuracy'])
		return model
