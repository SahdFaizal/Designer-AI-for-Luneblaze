import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Define the custom layer
class AttentionLayer(tf.keras.layers.Layer):
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

# Load the model with custom objects
def load_custom_model(model_path):
    custom_objects = {'AttentionLayer': AttentionLayer}
    model = load_model(model_path, custom_objects=custom_objects)
    return model

# Load the model
model = load_custom_model('rnn_with_attention_model.h5')
print("Model loaded from 'rnn_with_attention_model.h5'")

# Load tokenizer and label encoders
with open('Tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('LabelEncoder1.pkl', 'rb') as file:
    label_encoder1 = pickle.load(file)

with open('LabelEncoder2.pkl', 'rb') as file:
    label_encoder2 = pickle.load(file)

# Load input data
input_data = "Harmony Valley Academy (Rating: 9) Summary: Harmony Valley Academy boasts cutting-edge technology, a diverse curriculum tailored to individual student needs, and strong community involvement. Emphasizes a balanced approach to education with a focus on both academic excellence and character development. Provides a nurturing environment with a range of extracurricular activities to foster personal growth and social skills."

# Save input data as a PDF
def save_input_data_as_pdf(input_text, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y_position = height - 1 * inch  # Start position from top
    line_height = 14  # Line height
    c.setFont("Helvetica", 12)

    for line in input_text.split('\n'):
        if y_position <= 1 * inch:  # If space is running out, add a new page
            c.showPage()
            y_position = height - 1 * inch
            c.setFont("Helvetica", 12)
        
        c.drawString(1 * inch, y_position, line)
        y_position -= line_height

    c.save()

# Save the input data to a PDF file
input_pdf_path = 'Input_Data.pdf'
save_input_data_as_pdf(input_data, input_pdf_path)
print(f"Input data saved to '{input_pdf_path}'")

# Prepare the input data for prediction
input_sequences = tokenizer.texts_to_sequences([input_data])
input_sequences_padded = pad_sequences(input_sequences, padding='post')

# Predict
outputs = model.predict(input_sequences_padded)
output1_predictions, output2_predictions = outputs

# Decode predictions
num_classes1 = output1_predictions.shape[1]
num_classes2 = output2_predictions.shape[1]

output1_predictions_decoded = label_encoder1.inverse_transform(np.argmax(output1_predictions, axis=1))
output2_predictions_decoded = label_encoder2.inverse_transform(np.argmax(output2_predictions, axis=1))

def create_pdf(output_path, predictions):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    line_height = 14  # Line height
    max_lines_per_page = int((height - 2 * inch) / line_height)  # Calculate number of lines per page

    first_page = True
    for i, pred in enumerate(predictions):
        # Add a new page for each prediction, but only after the first page
        if not first_page:
            c.showPage()
        first_page = False

        y_position = height - 1 * inch  # Start position from top
        c.setFont("Helvetica", 12)
        for line in pred.split('\n'):
            if y_position <= 1 * inch:  # If space is running out, add a new page
                c.showPage()
                y_position = height - 1 * inch
                c.setFont("Helvetica", 12)

            c.drawString(1 * inch, y_position, line)
            y_position -= line_height

    c.save()

# Prepare predictions for PDF creation
output1_predictions_formatted = [f"{pred}" for pred in output1_predictions_decoded]
output2_predictions_formatted = [f"{pred}" for pred in output2_predictions_decoded]

# Save Output 1 predictions to PDF
create_pdf('Output1_Predictions.pdf', output1_predictions_formatted)
print("Output 1 predictions saved to 'Output1_Predictions.pdf'")

# Save Output 2 predictions to PDF
create_pdf('Output2_Predictions.pdf', output2_predictions_formatted)
print("Output 2 predictions saved to 'Output2_Predictions.pdf'")
