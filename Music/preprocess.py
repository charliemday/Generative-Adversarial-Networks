import glob
import numpy as np
from music21 import converter, instrument, note, chord

from keras.utils import np_utils


def get_notes(filepath):

    # Define empty note list
    notes = []

    for file in glob.glob("%s/*.mid" % filepath):
        # Convert file into a music21 object
        midi = converter.parse(file)
        print("Parsing %s" % file)

        notes_to_parse = None

        # Check if the file has instrumental parts
        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        # Is it a Note or a Chord?
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def prepare_sequences(notes):

    n_vocab = len(set(notes))

    # Define the sequence length
    sequence_length = 100

    # Get all the pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the network to be compatible with LSTMs
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab) / 2) / (float(n_vocab) / 2)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)
