from models import LstmModel
from models import CnnModel
from models import TextPredModel

import sys, types, time, random, os

import tensorflow_datasets as tfds
import tensorflow as tf

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def readCommand(args):
	return args

if __name__ == '__main__':
	args = readCommand( sys.argv[1:] )
	dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
	train_dataset, test_dataset = dataset['train'], dataset['test']
	encoder = info.features['text'].encoder

	train_dataset = train_dataset.shuffle(BUFFER_SIZE)
	train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
	test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

	sm = CnnModel(encoder, emd_dim=64)
	sm.train(train_dataset, test_dataset)
	print(sm.infer("How you doing?"))
	pass