# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:56:59 2024

@author: Enrico
"""
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import pandas as pd
import contractions
import re
import pickle
import random
import argparse

#tf.config.run_functions_eagerly(True)  # Enable eager execution
print(tf.executing_eagerly())  # Should print True if eager execution is on

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# For reproducibility.
np.random.seed(42)
tf.random.set_seed(42)


##### Helper Functions  ########################################################
## MASKING ####################################################################
# don't want the model to learn from the padding values
# The output of this function is a mask that identifies which elements of the sequence are padding (1.0), and which are valid elements (0.0).
def create_padding_mask(seq):
    #breakpoint()
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32) # creates a boolean mask where True means the token is padding (0), and False means it's a real token
    # Adding 2, 3 dimn using tf.newaxis, 2-> As this mask will be multiplied with each attention head and 3-> for each words (token) in a sentance
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len) the numbesr 1,1 enable broadcasting across the num_heads and seq_len

###############################################################################
# Looakahead mask

"""The look-ahead mask is used to mask the future tokens in a sequence. 
In other words, the mask indicates which entries should not be used.
"""
def create_look_ahead_mask(dec_inputs):
    #The look-ahead mask is used to mask the future tokens in a sequence
    #band_part with this setting creates lower triangular matrix that's why subtracting from 1
    # [[0., 1., 1.],
    #  [0., 0., 1.],
    #  [0., 0., 0.]] output with size:3

    """Mask future tokens in decoder inputs."""
    seq_len = tf.shape(dec_inputs)[1]  # Get the sequence length from the decoder inputs
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    #mask_numpy = tf.keras.backend.eval(mask) ## debug
    return mask  # (seq_len, seq_len)

# debug 
# x = tf.random.uniform((1, 3))
# temp = create_look_ahead_mask(x.shape[1])
# x_ = x.numpy()
# m = temp.numpy()
###############################################################################
# compute attention score taking dot product of the query (Q) and key (K) matrices. -->
# -->  matrix of scores which is then used to compute a weighted sum of the values (V).
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
    q: query shape == (..., seq_len_q, depth) # NOTE: depth=dk
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """ 
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk) # scaled by the square root of the dimensionality of the key
    # add the mask to the scaled tensor. 
    if mask is not None:
        scaled += (mask * -1e9)  # -1e9 ~ (-INFINITY) => where ever mask is set, make its scaled value close to -INF
    # elements close to negative infinity softmax, their attention weight becomes essentially 0
    # attention weight for a particular position is zero, the gradient with respect to that position also becomes zero
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled, axis=-1)  # (..., seq_len_q, seq_len_k) -->  how much focus each query token should have on each key token
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v) --->  new representation of the input sequence
    return output, attention_weights

#### POSITIONAL ENCODING ######################################################
# add information about the relative position of tokens in a sequence
## positional Encoding  'i' -> embedding dimn index, 'pos' -> word index in a sentence
# [pos, i]
# i ---> pos 0 , embedding matrix [0,0] [0,1] [0,2] [0,3] --> sin(pos/1000*exp(2i/d))   cos()  sin()  cos() -> 0 1 0 1
# am --> pos 1 , embedding matrix [1,0] [1,1] [1,2] [1,3] --> sin()   cos()  sin()  cos() --> 
# free --> pos 2 , embedding matrix [2,0] [2,1] [2,2] [2,3] --> sin()   cos()  sin()  cos()
class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs): 
    #step injects positional information into the input embeddings
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :] # slices the precomputed positional encodings to match the sequence length of the current batch


# position = 50  # Maximum sequence length
# d_model = 512  # Embedding dimension
# pos_encoding_layer = PositionalEncoding(position, d_model)
# # Create a dummy input: (batch_size, sequence_length, d_model)
# sample_input = tf.random.uniform((1, 50, 512))

# # Apply positional encoding
# encoded_output = pos_encoding_layer(sample_input)

# # Print the shape and inspect the result
# print(encoded_output.shape)  # (batch_size, sequence_length, d_model)
# print(encoded_output)  # Output the positional encoding result


# plt.pcolormesh(encoded_output[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
###############################################################################
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads # depth is the dimension of each attention head (d_model / num_heads).

    self.query_dense = tf.keras.layers.Dense(units=d_model) # (batch_size, seq_len, d_model
    self.key_dense = tf.keras.layers.Dense(units=d_model) # (batch_size, seq_len, d_model
    self.value_dense = tf.keras.layers.Dense(units=d_model) # (batch_size, seq_len, d_model

    self.dense = tf.keras.layers.Dense(units=d_model) # (batch_size, seq_len, d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query) # (batch_size, seq_len, d_model)
    key = self.key_dense(key) # (batch_size, seq_len, d_model)
    value = self.value_dense(value) # (batch_size, seq_len, d_model)

    # split heads
    query = self.split_heads(query, batch_size) # (batch_size, num_heads, seq_len_q, depth)
    key = self.split_heads(key, batch_size) # (batch_size, num_heads, seq_len_q, depth)
    value = self.split_heads(value, batch_size) # (batch_size, num_heads, seq_len_q, depth)
    
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
      # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights  = scaled_dot_product_attention(query, key, value, mask) # (batch_size, num_heads, seq_len, depth)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model)) #  flattens the num_heads and depth dimensions into one dimension.

    outputs = self.dense(concat_attention) # (batch_size, seq_len, d_model)
    return outputs, attention_weights

###############################################################################
### encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="encoder_layer", **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.units = units
        self.dropout = dropout

        # Multi-head attention layer: outputs (batch_size, seq_len, d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, name="attention")
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed-forward network: first dense layer (batch_size, seq_len, units), then (batch_size, seq_len, d_model)
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, padding_mask, training):
        # Multi-head attention output: (batch_size, seq_len, d_model)
        attention, _ = self.mha({
            'query': inputs,  # (batch_size, seq_len, d_model)
            'key': inputs,    # (batch_size, seq_len, d_model)
            'value': inputs,  # (batch_size, seq_len, d_model)
            'mask': padding_mask  # (batch_size, 1, 1, seq_len)
        })

        attention = self.dropout1(attention, training= training)  # (batch_size, seq_len, d_model)
        out1 = self.norm1(inputs + attention)  # (batch_size, seq_len, d_model)

        # Feed-forward network
        dense_output = self.dense1(out1)  # (batch_size, seq_len, units)
        dense_output = self.dense2(dense_output)  # (batch_size, seq_len, d_model)
        dense_output = self.dropout2(dense_output, training= training)  # (batch_size, seq_len, d_model)
        out2 = self.norm2(out1 + dense_output)  # (batch_size, seq_len, d_model)

        return out2
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="decoder_layer", **kwargs):
        super(DecoderLayer, self).__init__(name=name, **kwargs)

        # First multi-head attention layer: self-attention for decoder input
        self.mha1 = MultiHeadAttention(d_model, num_heads, name="attention_1")
        self.mha2 = MultiHeadAttention(d_model, num_heads, name="attention_2")
        
        # Dropout and normalization
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Second multi-head attention: cross-attention with encoder output
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed-forward network: first dense layer (batch_size, seq_len, units), then (batch_size, seq_len, d_model)
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask,training):
        # Self-attention: outputs (batch_size, seq_len, d_model)
        attention1, _ = self.mha1({
            'query': inputs,  # (batch_size, seq_len, d_model)
            'key': inputs,    # (batch_size, seq_len, d_model)
            'value': inputs,  # (batch_size, seq_len, d_model)
            'mask': look_ahead_mask  # (batch_size, 1, seq_len, seq_len) --> the mask is applied to the attention scores before the softmax step !
        })
        attention1 = self.dropout1( attention1, training=training)
        attention1 = self.norm1(attention1 + inputs)  # (batch_size, seq_len, d_model)

        # Cross-attention with encoder output
        attention2, _ = self.mha2({
            'query': attention1,    # (batch_size, seq_len, d_model)
            'key': enc_outputs,     # (batch_size, seq_len_enc, d_model)
            'value': enc_outputs,   # (batch_size, seq_len_enc, d_model)
            'mask': padding_mask    # (batch_size, 1, 1, seq_len_enc)
        })

        attention2 = self.dropout2(attention2, training=training)  # (batch_size, seq_len, d_model)
        out2 = self.norm2(attention2 + attention1)  # (batch_size, seq_len, d_model)

        # Feed-forward network
        dense_output = self.dense1(out2)  # (batch_size, seq_len, units)
        dense_output = self.dense2(dense_output)  # (batch_size, seq_len, d_model)
        dense_output = self.dropout3(dense_output, training= training)  # (batch_size, seq_len, d_model)
        out3 = self.norm3(out2 + dense_output)  # (batch_size, seq_len, d_model)

        return out3

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model

        # Embedding layer for input tokens (batch_size, seq_len, d_model)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        # Stack of encoder layers
        self.enc_layers = [EncoderLayer(units, d_model, num_heads, dropout, name=f"encoder_layer_{i}") for i in range(num_layers)]

    def call(self, inputs, padding_mask,training):
        embeddings = self.embedding(inputs)  # (batch_size, seq_len, d_model)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # scaling embeddings by the square root of the dimension of the model --> help training gradient calc
        embeddings = self.pos_encoding(embeddings)  # (batch_size, seq_len, d_model) -> sum btw pre calculated pos encoding and embedding
        outputs = self.dropout(embeddings, training=training)  # (batch_size, seq_len, d_model)

        # Pass through each encoder layer
        for i in range(self.num_layers):
            outputs = self.enc_layers[i](outputs, padding_mask, training)  # (batch_size, seq_len, d_model)

        return outputs  # (batch_size, seq_len, d_model)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model

        # Embedding layer for input tokens (batch_size, seq_len, d_model)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        # Stack of decoder layers
        self.dec_layers = [DecoderLayer(units, d_model, num_heads, dropout, name=f"decoder_layer_{i}") for i in range(num_layers)]

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask, training):
        embeddings = self.embedding(inputs)  # (batch_size, seq_len, d_model)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.pos_encoding(embeddings)  # (batch_size, seq_len, d_model)
        outputs = self.dropout(embeddings, training = training)  # (batch_size, seq_len, d_model)

        # Pass through each decoder layer
        for i in range(self.num_layers):
            outputs = self.dec_layers[i](outputs, enc_outputs, look_ahead_mask, padding_mask, training)  # (batch_size, seq_len, d_model)

        return outputs  # (batch_size, seq_len, d_model)
###############################################################################
class Transformer(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer", **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)
        
        self.encoder = Encoder(
            vocab_size=input_vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            vocab_size=target_vocab_size,
            num_layers=num_layers,
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, name="outputs")

    def create_padding_mask(self, inputs):
        """Mask all padding tokens."""
        return tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask'
        )(inputs)  # Mask shape: (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, dec_inputs):
        """Mask future tokens in decoder inputs."""
        return tf.keras.layers.Lambda(
            create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask'
        )(dec_inputs)  # Mask shape: (batch_size, 1, seq_len, seq_len)

    def call(self, inputs, dec_inputs, training=False):
        #create_masks ###
        # Encoder padding mask: (batch_size, 1, 1, seq_len)
        enc_padding_mask = self.create_padding_mask(inputs) #  mask is applied in the encoder's self-attention layers

        # Look-ahead mask for decoder: (batch_size, 1, seq_len, seq_len)
        look_ahead_mask = self.create_look_ahead_mask(dec_inputs)
        #mask_look_ahead_mask = tf.keras.backend.eval(look_ahead_mask)
        # Decoder padding mask for cross-attention: (batch_size, 1, 1, seq_len)
        dec_padding_mask = self.create_padding_mask(inputs)
        #mask_padding_mask = tf.keras.backend.eval(dec_padding_mask)[0] ## debug , 1 = padding , 0  are valid elements 
        ##################
        
        # Encoder outputs: (batch_size, seq_len, d_model)
        enc_outputs = self.encoder(inputs, enc_padding_mask, training)

        # Decoder outputs: (batch_size, seq_len, d_model)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask, training) # mask the padding tokens when the decoder attends to the encoder's output.

        # Final output layer: (batch_size, seq_len, vocab_size)
        outputs = self.final_layer(dec_outputs)

        return outputs




###############################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') # from_logits=True is specified, TensorFlow will apply the softmax function internally before calculating the cross-entrop
def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))  # Ignore padding tokens
    loss = loss_object(y_true, y_pred) # y_true[batch,seq_lenght] y_pred[batch, seq_lenght, voc_size]
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)
###############################################################################



#history = transformer_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
# Custom training step
# Metrics to track training and validation loss
train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
val_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

@tf.function(reduce_retracing=True) #  function is converted into a TensorFlow computational graph
def train_step(inputs,dec_inputs, target):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = transformer_model(inputs, dec_inputs, training=True)  # Pass training=True for Dropout
        # predictions = the model predicts a probability distribution over all N possible tokens in the vocabulary
        # Compute loss
        loss = loss_function(target, predictions)
    
    # Compute gradients with respect to trainable variables
    gradients = tape.gradient(loss, transformer_model.trainable_variables)
    
    # Apply gradients to optimizer
    optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))
    
    # Update the training loss metric
    train_loss_metric.update_state(loss)
    train_accuracy_metric.update_state(target, predictions)
    
    return loss

# Validation step (no gradient computation)
@tf.function (reduce_retracing=True)#  function is converted into a TensorFlow computational graph
def val_step(inputs, dec_inputs, target):
    # Forward pass during validation
    predictions = transformer_model(inputs, dec_inputs, training=False)
    
    # Compute loss
    loss = loss_function(target, predictions)
    
    # Update the validation loss metric
    val_loss_metric.update_state(loss)
    val_accuracy_metric.update_state(target, predictions)
    
    return loss

###############################################################################
### learning rate scheduling ##################################################
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def get_current_lr(optimizer):
    return optimizer._decayed_lr(tf.float32).numpy()  # This gets the learning rate after any decay/scheduling


###############################################################################
####### data loading - processing - generate token
# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Remove special characters (except alphanumeric and spaces)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text



def load_and_process_data(path_aq):
    data = pd.read_csv(path_aq)
    # Apply preprocessing to both Question and Answer columns
    data['Question'] = data['Question'].apply(preprocess_text)
    data['Answer'] = data['Answer'].apply(preprocess_text)

    train, validation = train_test_split(data, test_size=0.2, random_state=4)

    vocab_ans = list(set(" ".join(train['Answer'].values).split()))
    vocab_ques = list(set(" ".join(train['Question'].values).split()))
    vocab_size_ans, vocab_size_ques = len(vocab_ans), len(vocab_ques)

    print(f"vocab_size_ans, vocab_size_ques: {vocab_size_ans}, {vocab_size_ques}")

    # Train the SentencePiece model for answers
    temp_file_ans = 'train_corpus_answer.txt'
    with open(temp_file_ans, 'w', encoding='utf-8') as f:
        for sentence in train['Answer']:
            f.write(f"{sentence}\n")
    
    ### debug
    # # Step 1: Read the text file
    # with open('train_corpus_answer.txt', 'r', encoding='utf-8') as file:
    #     textdbg = file.read()
        
    # # Step 2: Clean the text
    # # Convert text to lowercase and remove punctuation
    # textdbg  = textdbg .lower()
    
    # # Step 3: Split the text into words
    # words = textdbg .split()
    # # Step 4: Use a set to get unique words
    # unique_words = set(words)
    
    # # Step 5: Count the number of unique words
    # vocabulary_size = len(unique_words)
    #  subword tokens in SentencePiece, which is likely smaller than the 12942 unique words because subword tokenization allows for reuse of the same token across multiple words
    spm.SentencePieceTrainer.train(input=temp_file_ans, model_prefix='tokenizer_a', vocab_size=9771, model_type='unigram')
    tokenizer_a = spm.SentencePieceProcessor(model_file='tokenizer_a.model')

    # Train the SentencePiece model for questions
    temp_file_ques = 'train_corpus_question.txt' ### temp file 
    with open(temp_file_ques, 'w', encoding='utf-8') as f:
        for sentence in train['Question']:
            f.write(f"{sentence}\n")

    spm.SentencePieceTrainer.train(input=temp_file_ques, model_prefix='tokenizer_q', vocab_size=5218, model_type='unigram')
    tokenizer_q = spm.SentencePieceProcessor(model_file='tokenizer_q.model')
    ### output !
    return {
        'train': train,
        'validation': validation,
        'tokenizer_a': tokenizer_a,
        'tokenizer_q': tokenizer_q
    }

### add start - end token 
def encode(ques, ans, tokenizer_q, tokenizer_a):
    ques = [tokenizer_q.vocab_size()] + tokenizer_q.encode(ques.numpy()) + [tokenizer_q.vocab_size()+1]
    ans = [tokenizer_a.vocab_size()] + tokenizer_a.encode(ans.numpy()) + [tokenizer_a.vocab_size()+1]
    return ques, ans

def tf_encode(ques, ans, tokenizer_q, tokenizer_a):
    result_ques, result_ans = tf.py_function(lambda q, a: encode(q, a, tokenizer_q, tokenizer_a), [ques, ans], [tf.int64, tf.int64])
    result_ques.set_shape([None])
    result_ans.set_shape([None])
    return result_ques, result_ans
######### evaluate functions ##################################################
# def create_masks(inp, tar):
#     # Encoder padding mask
#     enc_padding_mask = create_padding_mask(inp)

#     # Used in the 2nd attention block in the decoder.
#     # This padding mask is used to mask the encoder outputs.
#     dec_padding_mask = create_padding_mask(inp)

#     # Used in the 1st attention block in the decoder.
#     # It is used to pad and mask future tokens in the input received by 
#     # the decoder.
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#     return enc_padding_mask, combined_mask, dec_padding_mask


MAX_LENGTH = 350
def evaluate(inp_sentence, model, tokenizer_q, tokenizer_a):
    start_token = [tokenizer_q.vocab_size()]
    end_token = [tokenizer_q.vocab_size() + 1]

    # Add start and end token to the input question
    inp_sentence = start_token + tokenizer_q.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)  # add batch=1

    # Start token for decoder
    decoder_input = [tokenizer_a.vocab_size()]
    decoder_input = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        # Get predictions from model
        predictions = model(encoder_input, decoder_input, False)

        # Get the last token's prediction
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        # Get the token with the highest probability
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # Check if we hit the end token
        if tf.equal(predicted_id, tokenizer_a.vocab_size() + 1):
            print(f"=============\nGot end token\n=============")
            return tf.squeeze(decoder_input, axis=0)

        # Concatenate the predicted token to decoder_input
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    return tf.squeeze(decoder_input, axis=0)


def reply(sentence, transformer,  tokenizer_q, tokenizer_a):
    result = evaluate(sentence, transformer,  tokenizer_q, tokenizer_a)
    #breakpoint()
    # Convert the result tensor to a NumPy array
    result_array = result.numpy()
     
    # Get the vocabulary size once
    vocab_size = tokenizer_a.vocab_size()
     
    # Decode the predicted sentence
    predicted_sentence = tokenizer_a.Decode(
         [int(i) for i in result_array if i < vocab_size])
    
    return sentence, predicted_sentence


###############################################################################

# Main Process
if __name__ == "__main__":
    choice = 'eval'#'train'
    ############################################################################
    num_layers = 6 # number of encoder and decoder layers
    units = 512 #  dimensionality of the feedforward network (FFN) in the encoder and decoder layers. 
    d_model = 256 #  dimensionality of the embedding vectors
    num_heads = 8
    dropout = 0.1 #  0.1 means that 10% of the inputs to a given layer are set to zero
    train_dataset = val_dataset=  []
    EPOCHS = 200
    
    if (choice== 'train'):
        ##### data loadning and processing ########################################
        path_aq = './questions_and_answers_final.csv' # answer question path
    
        data = load_and_process_data(path_aq)
        train = data['train']
        validation = data['validation']
        tokenizer_q = data['tokenizer_q']
        tokenizer_a = data['tokenizer_a']
        ### save data ########################################################
        with open('data_tokenized/data_token.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # debug 
        # print(train['Question'].values[0], "\n", train['Answer'].values[0])
        # question, answer = tf_encode(train['Question'].values[0], train['Answer'].values[0], tokenizer_q, tokenizer_a)
        # print(question)
        # print(answer)
        
        # sample_string = 'encoder decoder' ## lower case
        # tokenized_string = tokenizer_a.Encode(sample_string)
        # print ('Tokenized string is {}'.format(tokenized_string))
        # original_string = tokenizer_a.Decode(tokenized_string)
        # print ('The original string: {}'.format(original_string))
        ###########################################################################
        
        # Example of how to instantiate and use the Transformer model:
        input_vocab_size = tokenizer_q.vocab_size() + 2
        target_vocab_size = tokenizer_a.vocab_size() + 2
  
    
        transformer_model = Transformer(input_vocab_size, target_vocab_size , num_layers, units, d_model, num_heads, dropout)

        # Compile the model
        # Define the optimizer
        #optimizer = tf.keras.optimizers.Adam()
        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,  epsilon=1e-9)
        transformer_model.compile(optimizer=optimizer,loss=loss_function,metrics=['accuracy'])
        ##########################################################
        ## Creating train_dataset/test_dataset
        # Step 1: Convert the training data (a dictionary) into a TensorFlow Dataset object
        train_dataset = tf.data.Dataset.from_tensor_slices(dict(train))
        # Step 2: Apply the `tf_encode` function to each element in the dataset
        # `tf_encode` adds start and end tokens to the 'Question' and 'Answer' and encodes them using the respective tokenizers `tokenizer_q` and `tokenizer_a`.
        train_dataset = train_dataset.map(lambda x:tf_encode(x['Question'], x['Answer'], tokenizer_q, tokenizer_a)) 
        # Step 3: Cache the dataset to speed up processing during training
        # Caching ensures that once the dataset is loaded and processed, it doesn't need to be recomputed in every epoch.
        train_dataset = train_dataset.cache()
        # Step 4: Shuffle the dataset with a buffer size of 20,000
        # A buffer size of 20,000 means it loads 20,000 elements into memory and shuffles them randomly.
        # Step 5: Batch the dataset and pad sequences to the length of the longest sequence in each batch
        # `[None], [None]` indicates that both the 'Question' and 'Answer' sequences will be padded independently based on the longest sequence in each batch
        train_dataset = train_dataset.shuffle(20000).padded_batch(64, padded_shapes=([None],[None])) 
        
        # Step 6: Prefetch batches for faster training
        # `prefetch` allows for loading the next batch of data while the current batch is being processed.
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) 
        
        val_dataset = tf.data.Dataset.from_tensor_slices(dict(validation))
        val_dataset = val_dataset.map(lambda x:tf_encode(x['Question'], x['Answer'], tokenizer_q, tokenizer_a))
        val_dataset = val_dataset.padded_batch(64, padded_shapes=([None],[None])) 
    
        ## debug
        # question, answer = next(iter(train_dataset))
        # print(question)  # To view the tokenized question tensor
        # print(answer)    # To view the tokenized answer tensor
    
        #########################################################
    
        # Etraining loop with validation
        save_checkpoint_every_n_step = 5
        # Directory to save checkpoints
        checkpoint_dir = './checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # Create a checkpoint object
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer_model)
        # Define a checkpoint manager to manage checkpoints
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
        
        # Restore the latest checkpoint if it exists
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        #### main loop ############################################################
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            
            # Reset the metrics at the start of each epoch
            train_loss_metric.reset_states()
            val_loss_metric.reset_states()
            train_accuracy_metric.reset_states()
            val_accuracy_metric.reset_states()
            
            # Training loop
            for (batch, (inputs,  target)) in enumerate(train_dataset):
                # Create dec_inputs and target
                dec_inputs = target[:, :-1]  # All tokens except the last --> end  --> decoder input
                target = target[:, 1:]       # All tokens except the first --> start --> prediction
                #breakpoint()
                loss = train_step(inputs,dec_inputs, target)
                # Get current learning rate
                current_lr = get_current_lr(optimizer)
                
                if batch % 20 == 0:
                    train_acc = train_accuracy_metric.result().numpy()
                    print(f"Epoch {epoch+1} Batch {batch} Training Loss: {loss.numpy():.4f}, Training Accuracy: {train_acc:.4f}, Learning Rate: {current_lr:.8f}")
                    #print(f"Epoch {epoch+1} Batch {batch} Training Loss: {loss.numpy():.4f}")
            
            # Validation loop
            for (batch, (val_inputs, val_target)) in enumerate(val_dataset):
                val_dec_inputs = val_target[:, :-1]  # Decoder input for validation
                val_target = val_target[:, 1:]       # Target for validation
                val_loss = val_step(val_inputs, val_dec_inputs, val_target)
            
            # After all batches, get the final training and validation loss for the epoch
            # After all batches, get the final training and validation loss and accuracy for the epoch
            train_loss = train_loss_metric.result()
            train_acc = train_accuracy_metric.result()
            val_loss = val_loss_metric.result()
            val_acc = val_accuracy_metric.result()
            
            # Get current learning rate at the end of the epoch
            current_lr = get_current_lr(optimizer)
            
            print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}, Learning Rate: {current_lr:.8f}")
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            
    
            # train_loss = train_loss_metric.result()
            # val_loss = val_loss_metric.result()
            
            # print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Save the model checkpoint 
            if (epoch + 1) % save_checkpoint_every_n_step == 0:
                saved_path = checkpoint_manager.save()
                print(f"Checkpoint saved at {saved_path}")
        
        
        # save weights
        transformer_model.save_weights('final_weights.h5')
        print("Weights saved to 'final_weights.h5'")
    
    elif(choice=='eval'):

        ### load data tokenized
        with open('data_tokenized/data_token.pickle', 'rb') as handle:
            data = pickle.load(handle)
        
        train = data['train']
        validation = data['validation']
        tokenizer_q = data['tokenizer_q']
        tokenizer_a = data['tokenizer_a']
        input_vocab_size = tokenizer_q.vocab_size() + 2
        target_vocab_size = tokenizer_a.vocab_size() + 2
        transformer_model = Transformer(input_vocab_size, target_vocab_size , num_layers, units, d_model, num_heads, dropout)
        
        # Run a dummy input through the model to create the variables
        dummy_input = tf.random.uniform((64, 27), dtype=tf.int64, minval=0, maxval=200)
        dummy_target = tf.random.uniform((64, 27), dtype=tf.int64, minval=0, maxval=200)
        dummy_out = transformer_model(dummy_input ,dummy_target, False)
        transformer_model.load_weights('final_weights.h5')
        
        test_q = validation["Question"]#.values[:10]
        test_a = validation["Answer"]#.values[:10]
        random_indices = random.sample(range(len(test_q)), 10)
        
        ####### test data #####################################################
        for i in random_indices: #range(20,25):
            input_test_sentence = test_q.values[i]
            input_sentence, pred_string = reply(input_test_sentence, transformer_model,  tokenizer_q, tokenizer_a)
            #breakpoint()
            print(f"{'-'*40}")
            print(f"Test Case {i+1}")
            print(f"{'-'*40}")
            print(f"Input      : {input_test_sentence}")
            print(f"Predicted  : {pred_string}")
            print(f"Actual     : {test_a.values[i]}")
            print(f"{'-'*40}\n")
        
        ### custom questions ###################################################
        qa_vocabulary = {
            "What is the name of Ron Weasley’s pet rat?": "Scabbers",
            "Describe Hagrid": "Hagrid is the half-giant Keeper of Keys and Grounds at Hogwarts, loyal to Dumbledore, and a friend to Harry.",
            "What is the name of the mirror that shows people their deepest desires?": "The Mirror of Erised",
            "What is the inscription on the Mirror of Erised and what does it mean?": "The inscription reads 'Erised stra ehru oyt ube cafru oyt on wohsi', meaning 'I show not your face but your heart’s desire' backwards.",
            "Why is Harry able to retrieve the Sorcerer’s Stone from the mirror at the end of the book?": "Harry is able to retrieve the Sorcerer’s Stone because he wanted to find it, but not use it for himself.",
            "What happens to Dudley during the visit to the zoo with Harry?": "Dudley falls into a snake enclosure after the glass disappears.",
            "What does Hagrid give Harry on Christmas Day?": "Hagrid gives Harry a wooden flute.",
            "What creature does Hagrid win in a card game that later becomes a key part of protecting the Sorcerer’s Stone?": "A dragon egg that hatches into a Norwegian Ridgeback named Norbert.",
            "What is the name of the girl who becomes friends with Harry and Ron?": "Hermione Granger"
                    }
        #breakpoint()
        for cq, actual_answer in qa_vocabulary.items():
            ### processing the string 
            cq = preprocess_text(cq)
            input_sentence, pred_string = reply(cq, transformer_model,  tokenizer_q, tokenizer_a)
            print(f"{'-'*40}")
            print(f"Test Case Custom")
            print(f"{'-'*40}")
            print(f"Input      : {cq}")
            print(f"Predicted  : {pred_string}")
            print(f"Actual     : {actual_answer}")
            print(f"{'-'*40}\n")

    
