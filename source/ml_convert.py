import sys
import os
import coremltools


# Convert a caffe model to a classifier in Core ML
coreml_model = coremltools.converters.keras.convert(('./10_000_twohidden/model.json', './10_000_twohidden/model.h5'))


# Now save the model
coreml_model.save('LetterClass.mlmodel')