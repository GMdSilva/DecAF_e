from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

from decaf_e.attentionLayer import AttentionLayer


class ModelBuilder:
	"""
	Class to build a Keras model with configurable parameters.
	"""

	def __init__(self, seq_len: int, dense_layers: int = 200, dropout_rate: float = 0.1):
		"""
		Initialize the ModelBuilder with the desired configuration.

		Args:
		seq_len (int): Length of the input sequences.
		dense_layers (int): Number of neurons in the dense layer.
		dropout_rate (float): Dropout rate for regularization.
		"""
		self.seq_len = seq_len
		self.dense_layers = dense_layers
		self.dropout_rate = dropout_rate

	def create_model(self) -> Sequential:
		"""
		Creates and compiles a Keras model with an attention layer and additional layers for binary classification.

		Returns:
		Sequential: A compiled Keras model ready for training.
		"""
		model = Sequential([
			AttentionLayer(input_shape=(self.seq_len, 23)),  # Assuming 23 represents features like amino acids
			Flatten(),
			Dense(self.dense_layers, activation='relu'),
			Dropout(self.dropout_rate),
			Dense(1, activation='sigmoid')
		])
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		return model
