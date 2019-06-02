import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
# from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn import metrics

def makeNet(imx, imy, chanCount, depth, startFilters, classCount):
    input = Input((imx,imy,chanCount))
    filters = startFilters
    chanDim = -1
    mod = input
    for i in range(depth):
        mod = Conv2D(filters, 3, name='ConvA'+str(i), padding='same')(mod)
        mod = Conv2D(filters, 3, name='ConvB'+str(i), padding='same')(mod)
        mod = Activation('relu')(mod)
        mod = BatchNormalization(axis=chanDim, name='BN'+str(i))(mod)
        mod = MaxPooling2D(name='Pool'+str(i))(mod)
        filters *= 2
    mod = Flatten(name='Flat')(mod)
    mod = Dense(64, name='DensA')(mod)
    mod = Activation("relu")(mod)
    mod = BatchNormalization(axis=chanDim, name='BN')(mod)
    mod = Dropout(rate=0.5)(mod)
    mod = Dense(32, name='DensB')(mod)
    mod = Activation("relu")(mod)
    mod = Dense(classCount, activation="softmax")(mod)
    return Model(input, mod)


# The digits dataset
digits = datasets.load_digits()
count = len(digits.images)
print('Loaded %d digit images' % (count))

allPics = digits.images[:, :, :, np.newaxis]
outcome = to_categorical(digits.target)
outCount = outcome.shape[0]
classCount = outcome.shape[1]

imCount = allPics.shape[0]
imx = allPics.shape[1]
imy = allPics.shape[2]
chanCount = allPics.shape[3]

print('Using %d pictures, %d by %d, %d channels' % (imCount, imx, imy, chanCount))
print('Outcome matrix is %d x %d' % (outCount, classCount))

# Model and optimization parameters
startFilters = 16
#depth = 2
depth = 3
bs = 32
epochs = 10
logdir = 'c:/temp'

# Make CNN model
model = makeNet(imx, imy, chanCount, depth, startFilters, classCount)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="categorical_crossentropy", optimizer=opt)

# Split data into training and test
split = train_test_split(allPics, outcome, test_size=0.25, random_state=42)
(trainPics, testPics, trainOut, testOut) = split

# Training
print("Training model...")
if True:
    model.fit(trainPics, trainOut, validation_data=(testPics, testOut), 
    epochs=epochs, batch_size=bs, callbacks=[TensorBoard(log_dir=logdir)])
else:
    # Prepare Data Augmentation
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, 
    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)
    datagen.fit(trainPics)
    # And then optimize using Data Augmentation
    model.fit_generator(datagen.flow(trainPics, trainOut, batch_size=bs),
                    steps_per_epoch=len(trainOut) / bs, epochs=epochs,
                    validation_data=(testPics, testOut),
                    callbacks=[TensorBoard(log_dir=logdir)])


# Validation
print("Validating model...")
pred = model.predict(testPics)
predOneHot = np.round(pred) # sloppy!

predLabel = np.argmax(pred, axis=1)
testLabel = np.argmax(testOut, axis=1)

acc = metrics.accuracy_score(testOut, predOneHot)
print('Classification accuracy on test set (onehot): %.3f' % acc)

acc = metrics.accuracy_score(testLabel, predLabel)
print('Classification accuracy on test set (label): %.3f' % acc)


print(metrics.confusion_matrix(testLabel, predLabel))

print('Now run tensorboard --logdir "' + logdir + '" and see localhost:6006 on browser')
