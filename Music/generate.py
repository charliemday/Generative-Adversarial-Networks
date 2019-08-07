import tensorflow as tf
from models import Models
from preprocess import *
from create import *

generator_weight = "weights/generator_model_500.h5"

generator = Models(rows=100).generator()

generator.load_weights(generator_weight)

def load_notes(filepath="testdata"):
    # Load dataset
    notes = get_notes(filepath)
    # Preprocess notes
    x_train, y_train = prepare_sequences(notes)

    return x_train, notes  # y_train may not be needed

dataset, notes = load_notes("Pokemon MIDIs")

generate(notes, generator)
