import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
	"""
	Custom Keras layer to apply attention mechanism on input features.

	This layer computes a weighted sum of the input features based on
	learned attention weights, enabling the model to focus more on relevant features.
	"""

	def __init__(self, **kwargs):
		"""
		Initialize the layer, this function sets up any attributes required by the layer.
		"""
		super(AttentionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		"""
		Create the weights of the layer. This function is called once, and is where the
		layer's trainable weights are defined.

		Args:
		input_shape (tuple): The shape of the input data.
		"""
		# Weight matrix for creating attention scores
		self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
		# Bias to help with the learning of the attention scores
		self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
		super(AttentionLayer, self).build(input_shape)

	def call(self, inputs):
		"""
		Call logic for the layer, where the actual computations are done.

		Args:
		inputs (tensor): Input tensor to the layer.

		Returns:
		tensor: Output tensor after applying the attention mechanism.
		"""
		# Compute the attention scores
		e = K.tanh(K.dot(inputs, self.W) + self.b)
		# Remove the last singleton dimension
		e = K.squeeze(e, axis=-1)
		# Compute the attention weights using softmax
		alpha = K.softmax(e)
		# Reshape alpha to make it suitable for element-wise multiplication
		alpha = K.expand_dims(alpha, -1)
		# Apply the attention weights to the input
		return inputs * alpha

	def compute_output_shape(self, input_shape):
		"""
		Compute the output shape of the layer based on the input shape.

		Args:
		input_shape (tuple): Shape tuple of the input.

		Returns:
		tuple: The same as the input shape, since the attention mechanism preserves dimensions.
		"""
		return input_shape
