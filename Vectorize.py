import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense, Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load and prepare data
with open('Input_Data.pkl', 'rb') as file:
    input_data = pickle.load(file)
with open('Output_Data1.pkl', 'rb') as file:
    output_data1 = pickle.load(file)
with open('Output_Data2.pkl', 'rb') as file:
    output_data2 = pickle.load(file)

# Tokenize and prepare sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_data)

num_words = len(tokenizer.word_index) + 1
input_sequences = tokenizer.texts_to_sequences(input_data)
input_sequences = pad_sequences(input_sequences, padding='post')

# Convert text labels to one-hot encoded labels
label_encoder1 = LabelEncoder()
integer_encoded1 = label_encoder1.fit_transform(output_data1)
integer_encoded1 = integer_encoded1.reshape(-1, 1)

onehot_encoder1 = OneHotEncoder(sparse_output=False)
output_data1 = onehot_encoder1.fit_transform(integer_encoded1)

label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(output_data2)
integer_encoded2 = integer_encoded2.reshape(-1, 1)

onehot_encoder2 = OneHotEncoder(sparse_output=False)
output_data2 = onehot_encoder2.fit_transform(integer_encoded2)

# Save label encoders
with open('LabelEncoder1.pkl', 'wb') as file:
    pickle.dump(label_encoder1, file)
with open('LabelEncoder2.pkl', 'wb') as file:
    pickle.dump(label_encoder2, file)

# Define custom attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer='random_normal',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_scores = tf.reduce_sum(inputs * self.W, axis=-1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def create_model(num_classes1, num_classes2):
    inputs = Input(shape=(None,), name='inputs')
    embedding = Embedding(input_dim=num_words, output_dim=256)(inputs)
    rnn = SimpleRNN(256, return_sequences=True)(embedding)
    context_vector = AttentionLayer()(rnn)
    
    # Define two output layers
    outputs1 = Dense(num_classes1, activation='softmax', name='output1')(context_vector)
    outputs2 = Dense(num_classes2, activation='softmax', name='output2')(context_vector)
    
    model = Model(inputs, [outputs1, outputs2])
    return model

# Define and compile the model
num_classes1 = output_data1.shape[1]  # Number of classes for output1
num_classes2 = output_data2.shape[1]  # Number of classes for output2

model = create_model(num_classes1, num_classes2)
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, 
              loss={'output1': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
              metrics={'output1': 'accuracy', 'output2': 'accuracy'})

# Train the model
history = model.fit(
    input_sequences,
    {'output1': output_data1, 'output2': output_data2},
    epochs=500,
    batch_size=1,
)

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('rnn_with_attention_model.h5')
print("Model saved to 'rnn_with_attention_model.h5'")

# Save tokenizer
with open('Tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
