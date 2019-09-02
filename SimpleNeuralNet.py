import sys

print(sys.version)

# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from ann_visualizer.visualize import ann_viz;

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = []
Y = []
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

ann_viz(model, title="My First Model")
