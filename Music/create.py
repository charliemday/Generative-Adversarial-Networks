import numpy as np
from music21 import stream, note, instrument, chord

def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []

    # Create note and chord objects
    for item in prediction_output:
        pattern = str(item[0])
        # If pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                # new_note.sortedInstrument = instrument.Piano()
                new_note.sortedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Else if pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            # new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes don't stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="samples/{}.mid".format(filename))


def generate(input_notes, generator, latent_dim = 1000):

    # Get notes
    notes = input_notes

    # Sort notes
    pitchnames = sorted(set(item for item in notes))

    # Integer to note (Note to integer?)
    int_to_note = dict((number, note)
                       for number, note in enumerate(pitchnames))

    # Create random noise
    noise = np.random.normal(0, 1, (1, latent_dim))

    # Predict 100 notes with the generator
    predictions = generator.predict(noise)

    # Get the normalization range for between -1 and 1
    norm_range = len(int_to_note) / 2

    # Convert predicted notes to index (normalize)
    pred_notes = [(x * norm_range + norm_range) for x in predictions[0]]

    pred_notes = [int_to_note[int(x)] for x in pred_notes]
    # pred_notes = [x for x in pred_notes]

    # Create a midi sample
    create_midi(pred_notes, filename = "final_weights_sample_02_08_19")
