# FER-Project
My Final Year Project improving the accuracy on the dataset by using Deep Learning Models

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.applications import VGG16, InceptionResNetV2
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax

from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau

import datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model
[2]
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive

!unzip /content/drive/MyDrive/fer.zip -d dataset
Loading and pre-processing dataset

[4]
train_dir = '/content/dataset/train'
test_dir = '/content/dataset/test'
[5]
datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, horizontal_flip=True)
[6]
datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1./255)
[7]
# Preprocess all test images
train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(48, 48),
                                              batch_size=64,
                                              color_mode="grayscale",
                                              class_mode='categorical')
Found 28709 images belonging to 7 classes.

[18]
test_generator = test_gen.flow_from_directory(test_dir,
                                              target_size=(48, 48),
                                              batch_size=64,
                                              color_mode="grayscale",
                                              shuffle = False,
                                              class_mode='categorical')
Found 7178 images belonging to 7 classes.

Defining the structure of model

[9]
def get_model(input_size, classes=7):
     #Initialising the CNN
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape =input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    #Compliling the model
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
[10]
fernet = get_model((48,48,1), 7)
fernet.summary()
WARNING:absl:`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 48, 48, 32)        320       
                                                                 
 conv2d_1 (Conv2D)           (None, 48, 48, 64)        18496     
                                                                 
 batch_normalization (BatchN  (None, 48, 48, 64)       256       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 24, 24, 64)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 24, 24, 64)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 24, 24, 128)       73856     
                                                                 
 conv2d_3 (Conv2D)           (None, 22, 22, 256)       295168    
                                                                 
 batch_normalization_1 (Batc  (None, 22, 22, 256)      1024      
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 11, 11, 256)      0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 11, 11, 256)       0         
                                                                 
 flatten (Flatten)           (None, 30976)             0         
                                                                 
 dense (Dense)               (None, 1024)              31720448  
                                                                 
 dropout_2 (Dropout)         (None, 1024)              0         
                                                                 
 dense_1 (Dense)             (None, 7)                 7175      
                                                                 
=================================================================
Total params: 32,116,743
Trainable params: 32,116,103
Non-trainable params: 640
_________________________________________________________________

[11]
chk_path = '/content/drive/MyDrive/emotions/ferNet.h5'
log_dir = "/content/drive/MyDrive/emotions/checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")



earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=6,
                              verbose=1,
                              min_delta=0.0001)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = CSVLogger('training.log')

callbacks = [reduce_lr]
Training the model

[12]
steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = test_generator.n // test_generator.batch_size

history = fernet.fit(x=train_generator,
                 validation_data= test_generator,
                 epochs=80,
                 callbacks=callbacks,
                 steps_per_epoch=steps_per_epoch,
                 validation_steps=validation_steps)
Epoch 1/80
448/448 [==============================] - 47s 75ms/step - loss: 5.0555 - accuracy: 0.2501 - val_loss: 50.2182 - val_accuracy: 0.1720 - lr: 0.0010
Epoch 2/80
448/448 [==============================] - 35s 78ms/step - loss: 2.0610 - accuracy: 0.2952 - val_loss: 1.8702 - val_accuracy: 0.3203 - lr: 0.0010
Epoch 3/80
448/448 [==============================] - 37s 82ms/step - loss: 1.7863 - accuracy: 0.3250 - val_loss: 1.8928 - val_accuracy: 0.3217 - lr: 0.0010
Epoch 4/80
448/448 [==============================] - 33s 73ms/step - loss: 1.6946 - accuracy: 0.3541 - val_loss: 1.7189 - val_accuracy: 0.3736 - lr: 0.0010
Epoch 5/80
448/448 [==============================] - 35s 78ms/step - loss: 1.6420 - accuracy: 0.3780 - val_loss: 1.5457 - val_accuracy: 0.4192 - lr: 0.0010
Epoch 6/80
448/448 [==============================] - 36s 81ms/step - loss: 1.6021 - accuracy: 0.3931 - val_loss: 1.4871 - val_accuracy: 0.4346 - lr: 0.0010
Epoch 7/80
448/448 [==============================] - 36s 79ms/step - loss: 1.5652 - accuracy: 0.4101 - val_loss: 1.4297 - val_accuracy: 0.4580 - lr: 0.0010
Epoch 8/80
448/448 [==============================] - 35s 79ms/step - loss: 1.5409 - accuracy: 0.4227 - val_loss: 1.3801 - val_accuracy: 0.4848 - lr: 0.0010
Epoch 9/80
448/448 [==============================] - 33s 74ms/step - loss: 1.5089 - accuracy: 0.4353 - val_loss: 1.4325 - val_accuracy: 0.4594 - lr: 0.0010
Epoch 10/80
448/448 [==============================] - 33s 73ms/step - loss: 1.4911 - accuracy: 0.4447 - val_loss: 1.4681 - val_accuracy: 0.4668 - lr: 0.0010
Epoch 11/80
448/448 [==============================] - 35s 78ms/step - loss: 1.4760 - accuracy: 0.4485 - val_loss: 1.3692 - val_accuracy: 0.4975 - lr: 0.0010
Epoch 12/80
448/448 [==============================] - 38s 84ms/step - loss: 1.4648 - accuracy: 0.4577 - val_loss: 1.3421 - val_accuracy: 0.5075 - lr: 0.0010
Epoch 13/80
448/448 [==============================] - 35s 78ms/step - loss: 1.4431 - accuracy: 0.4676 - val_loss: 1.3320 - val_accuracy: 0.5109 - lr: 0.0010
Epoch 14/80
448/448 [==============================] - 34s 76ms/step - loss: 1.4329 - accuracy: 0.4707 - val_loss: 1.2942 - val_accuracy: 0.5250 - lr: 0.0010
Epoch 15/80
448/448 [==============================] - 35s 77ms/step - loss: 1.4263 - accuracy: 0.4729 - val_loss: 1.3077 - val_accuracy: 0.5212 - lr: 0.0010
Epoch 16/80
448/448 [==============================] - 36s 80ms/step - loss: 1.4124 - accuracy: 0.4830 - val_loss: 1.2878 - val_accuracy: 0.5317 - lr: 0.0010
Epoch 17/80
448/448 [==============================] - 33s 73ms/step - loss: 1.3920 - accuracy: 0.4855 - val_loss: 1.3209 - val_accuracy: 0.5077 - lr: 0.0010
Epoch 18/80
448/448 [==============================] - 35s 77ms/step - loss: 1.3793 - accuracy: 0.4906 - val_loss: 1.2526 - val_accuracy: 0.5402 - lr: 0.0010
Epoch 19/80
448/448 [==============================] - 35s 78ms/step - loss: 1.3738 - accuracy: 0.4937 - val_loss: 1.4621 - val_accuracy: 0.4770 - lr: 0.0010
Epoch 20/80
448/448 [==============================] - 33s 72ms/step - loss: 1.3637 - accuracy: 0.4984 - val_loss: 1.2268 - val_accuracy: 0.5552 - lr: 0.0010
Epoch 21/80
448/448 [==============================] - 37s 82ms/step - loss: 1.3466 - accuracy: 0.5063 - val_loss: 1.2729 - val_accuracy: 0.5251 - lr: 0.0010
Epoch 22/80
448/448 [==============================] - 33s 73ms/step - loss: 1.3353 - accuracy: 0.5087 - val_loss: 1.2195 - val_accuracy: 0.5559 - lr: 0.0010
Epoch 23/80
448/448 [==============================] - 35s 78ms/step - loss: 1.3396 - accuracy: 0.5113 - val_loss: 1.2009 - val_accuracy: 0.5601 - lr: 0.0010
Epoch 24/80
448/448 [==============================] - 35s 78ms/step - loss: 1.3281 - accuracy: 0.5135 - val_loss: 1.2031 - val_accuracy: 0.5629 - lr: 0.0010
Epoch 25/80
448/448 [==============================] - 36s 81ms/step - loss: 1.3109 - accuracy: 0.5198 - val_loss: 1.2438 - val_accuracy: 0.5589 - lr: 0.0010
Epoch 26/80
448/448 [==============================] - 35s 79ms/step - loss: 1.3098 - accuracy: 0.5231 - val_loss: 1.2330 - val_accuracy: 0.5495 - lr: 0.0010
Epoch 27/80
448/448 [==============================] - 33s 73ms/step - loss: 1.2960 - accuracy: 0.5275 - val_loss: 1.3106 - val_accuracy: 0.5257 - lr: 0.0010
Epoch 28/80
448/448 [==============================] - 35s 78ms/step - loss: 1.2849 - accuracy: 0.5331 - val_loss: 1.1800 - val_accuracy: 0.5745 - lr: 0.0010
Epoch 29/80
448/448 [==============================] - 33s 74ms/step - loss: 1.2847 - accuracy: 0.5342 - val_loss: 1.2262 - val_accuracy: 0.5656 - lr: 0.0010
Epoch 30/80
448/448 [==============================] - 35s 78ms/step - loss: 1.2731 - accuracy: 0.5365 - val_loss: 1.2240 - val_accuracy: 0.5665 - lr: 0.0010
Epoch 31/80
448/448 [==============================] - 33s 73ms/step - loss: 1.2657 - accuracy: 0.5407 - val_loss: 1.1972 - val_accuracy: 0.5677 - lr: 0.0010
Epoch 32/80
448/448 [==============================] - 35s 77ms/step - loss: 1.2647 - accuracy: 0.5435 - val_loss: 1.2431 - val_accuracy: 0.5441 - lr: 0.0010
Epoch 33/80
448/448 [==============================] - 35s 78ms/step - loss: 1.2639 - accuracy: 0.5434 - val_loss: 1.1768 - val_accuracy: 0.5799 - lr: 0.0010
Epoch 34/80
448/448 [==============================] - 36s 81ms/step - loss: 1.2471 - accuracy: 0.5499 - val_loss: 1.1858 - val_accuracy: 0.5665 - lr: 0.0010
Epoch 35/80
448/448 [==============================] - 33s 73ms/step - loss: 1.2531 - accuracy: 0.5476 - val_loss: 1.1495 - val_accuracy: 0.5891 - lr: 0.0010
Epoch 36/80
448/448 [==============================] - 35s 77ms/step - loss: 1.2360 - accuracy: 0.5530 - val_loss: 1.1960 - val_accuracy: 0.5734 - lr: 0.0010
Epoch 37/80
448/448 [==============================] - 35s 78ms/step - loss: 1.2347 - accuracy: 0.5564 - val_loss: 1.1805 - val_accuracy: 0.5717 - lr: 0.0010
Epoch 38/80
448/448 [==============================] - 34s 75ms/step - loss: 1.2255 - accuracy: 0.5602 - val_loss: 1.1749 - val_accuracy: 0.5748 - lr: 0.0010
Epoch 39/80
448/448 [==============================] - 35s 77ms/step - loss: 1.2251 - accuracy: 0.5580 - val_loss: 1.2004 - val_accuracy: 0.5709 - lr: 0.0010
Epoch 40/80
448/448 [==============================] - 32s 71ms/step - loss: 1.2267 - accuracy: 0.5582 - val_loss: 1.1795 - val_accuracy: 0.5812 - lr: 0.0010
Epoch 41/80
448/448 [==============================] - ETA: 0s - loss: 1.2162 - accuracy: 0.5611
Epoch 41: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
448/448 [==============================] - 35s 78ms/step - loss: 1.2162 - accuracy: 0.5611 - val_loss: 1.1723 - val_accuracy: 0.5830 - lr: 0.0010
Epoch 42/80
448/448 [==============================] - 35s 78ms/step - loss: 1.1446 - accuracy: 0.5898 - val_loss: 1.1152 - val_accuracy: 0.6028 - lr: 2.0000e-04
Epoch 43/80
448/448 [==============================] - 34s 76ms/step - loss: 1.1107 - accuracy: 0.5998 - val_loss: 1.1074 - val_accuracy: 0.6049 - lr: 2.0000e-04
Epoch 44/80
448/448 [==============================] - 35s 78ms/step - loss: 1.0973 - accuracy: 0.6039 - val_loss: 1.0909 - val_accuracy: 0.6067 - lr: 2.0000e-04
Epoch 45/80
448/448 [==============================] - 35s 78ms/step - loss: 1.0910 - accuracy: 0.6069 - val_loss: 1.0887 - val_accuracy: 0.6035 - lr: 2.0000e-04
Epoch 46/80
448/448 [==============================] - 34s 76ms/step - loss: 1.0828 - accuracy: 0.6094 - val_loss: 1.0748 - val_accuracy: 0.6115 - lr: 2.0000e-04
Epoch 47/80
448/448 [==============================] - 36s 80ms/step - loss: 1.0778 - accuracy: 0.6123 - val_loss: 1.0924 - val_accuracy: 0.6057 - lr: 2.0000e-04
Epoch 48/80
448/448 [==============================] - 35s 79ms/step - loss: 1.0666 - accuracy: 0.6132 - val_loss: 1.0740 - val_accuracy: 0.6127 - lr: 2.0000e-04
Epoch 49/80
448/448 [==============================] - 35s 79ms/step - loss: 1.0665 - accuracy: 0.6165 - val_loss: 1.0672 - val_accuracy: 0.6110 - lr: 2.0000e-04
Epoch 50/80
448/448 [==============================] - 35s 79ms/step - loss: 1.0606 - accuracy: 0.6135 - val_loss: 1.0961 - val_accuracy: 0.6018 - lr: 2.0000e-04
Epoch 51/80
448/448 [==============================] - 35s 77ms/step - loss: 1.0554 - accuracy: 0.6164 - val_loss: 1.0698 - val_accuracy: 0.6126 - lr: 2.0000e-04
Epoch 52/80
448/448 [==============================] - 36s 80ms/step - loss: 1.0502 - accuracy: 0.6189 - val_loss: 1.0915 - val_accuracy: 0.6131 - lr: 2.0000e-04
Epoch 53/80
448/448 [==============================] - 35s 77ms/step - loss: 1.0432 - accuracy: 0.6199 - val_loss: 1.0814 - val_accuracy: 0.6131 - lr: 2.0000e-04
Epoch 54/80
448/448 [==============================] - 34s 77ms/step - loss: 1.0349 - accuracy: 0.6270 - val_loss: 1.1141 - val_accuracy: 0.6097 - lr: 2.0000e-04
Epoch 55/80
448/448 [==============================] - ETA: 0s - loss: 1.0421 - accuracy: 0.6218
Epoch 55: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
448/448 [==============================] - 38s 84ms/step - loss: 1.0421 - accuracy: 0.6218 - val_loss: 1.0749 - val_accuracy: 0.6204 - lr: 2.0000e-04
Epoch 56/80
448/448 [==============================] - 36s 80ms/step - loss: 1.0167 - accuracy: 0.6341 - val_loss: 1.0583 - val_accuracy: 0.6189 - lr: 4.0000e-05
Epoch 57/80
448/448 [==============================] - 36s 80ms/step - loss: 1.0140 - accuracy: 0.6309 - val_loss: 1.0605 - val_accuracy: 0.6169 - lr: 4.0000e-05
Epoch 58/80
448/448 [==============================] - 35s 79ms/step - loss: 1.0080 - accuracy: 0.6330 - val_loss: 1.0575 - val_accuracy: 0.6151 - lr: 4.0000e-05
Epoch 59/80
448/448 [==============================] - 37s 83ms/step - loss: 1.0050 - accuracy: 0.6348 - val_loss: 1.0610 - val_accuracy: 0.6158 - lr: 4.0000e-05
Epoch 60/80
448/448 [==============================] - 35s 78ms/step - loss: 1.0068 - accuracy: 0.6357 - val_loss: 1.0531 - val_accuracy: 0.6205 - lr: 4.0000e-05
Epoch 61/80
448/448 [==============================] - 35s 79ms/step - loss: 1.0056 - accuracy: 0.6376 - val_loss: 1.0582 - val_accuracy: 0.6177 - lr: 4.0000e-05
Epoch 62/80
448/448 [==============================] - 34s 76ms/step - loss: 1.0020 - accuracy: 0.6363 - val_loss: 1.0588 - val_accuracy: 0.6204 - lr: 4.0000e-05
Epoch 63/80
448/448 [==============================] - 37s 83ms/step - loss: 0.9995 - accuracy: 0.6355 - val_loss: 1.0513 - val_accuracy: 0.6208 - lr: 4.0000e-05
Epoch 64/80
448/448 [==============================] - 34s 75ms/step - loss: 0.9979 - accuracy: 0.6395 - val_loss: 1.0616 - val_accuracy: 0.6183 - lr: 4.0000e-05
Epoch 65/80
448/448 [==============================] - 34s 75ms/step - loss: 0.9968 - accuracy: 0.6399 - val_loss: 1.0505 - val_accuracy: 0.6200 - lr: 4.0000e-05
Epoch 66/80
448/448 [==============================] - 34s 76ms/step - loss: 0.9995 - accuracy: 0.6388 - val_loss: 1.0555 - val_accuracy: 0.6201 - lr: 4.0000e-05
Epoch 67/80
448/448 [==============================] - 35s 79ms/step - loss: 0.9993 - accuracy: 0.6382 - val_loss: 1.0547 - val_accuracy: 0.6233 - lr: 4.0000e-05
Epoch 68/80
448/448 [==============================] - 35s 79ms/step - loss: 0.9986 - accuracy: 0.6372 - val_loss: 1.0583 - val_accuracy: 0.6189 - lr: 4.0000e-05
Epoch 69/80
448/448 [==============================] - 34s 76ms/step - loss: 0.9839 - accuracy: 0.6449 - val_loss: 1.0527 - val_accuracy: 0.6225 - lr: 4.0000e-05
Epoch 70/80
448/448 [==============================] - 33s 74ms/step - loss: 0.9910 - accuracy: 0.6416 - val_loss: 1.0564 - val_accuracy: 0.6226 - lr: 4.0000e-05
Epoch 71/80
448/448 [==============================] - ETA: 0s - loss: 0.9853 - accuracy: 0.6443
Epoch 71: ReduceLROnPlateau reducing learning rate to 8.000000525498762e-06.
448/448 [==============================] - 37s 82ms/step - loss: 0.9853 - accuracy: 0.6443 - val_loss: 1.0508 - val_accuracy: 0.6208 - lr: 4.0000e-05
Epoch 72/80
448/448 [==============================] - 35s 79ms/step - loss: 0.9889 - accuracy: 0.6417 - val_loss: 1.0495 - val_accuracy: 0.6219 - lr: 8.0000e-06
Epoch 73/80
448/448 [==============================] - 33s 73ms/step - loss: 0.9852 - accuracy: 0.6448 - val_loss: 1.0507 - val_accuracy: 0.6225 - lr: 8.0000e-06
Epoch 74/80
448/448 [==============================] - 34s 77ms/step - loss: 0.9808 - accuracy: 0.6426 - val_loss: 1.0508 - val_accuracy: 0.6237 - lr: 8.0000e-06
Epoch 75/80
448/448 [==============================] - 34s 76ms/step - loss: 0.9923 - accuracy: 0.6403 - val_loss: 1.0496 - val_accuracy: 0.6242 - lr: 8.0000e-06
Epoch 76/80
448/448 [==============================] - 35s 78ms/step - loss: 0.9873 - accuracy: 0.6430 - val_loss: 1.0476 - val_accuracy: 0.6215 - lr: 8.0000e-06
Epoch 77/80
448/448 [==============================] - 35s 77ms/step - loss: 0.9847 - accuracy: 0.6429 - val_loss: 1.0498 - val_accuracy: 0.6242 - lr: 8.0000e-06
Epoch 78/80
448/448 [==============================] - 32s 72ms/step - loss: 0.9807 - accuracy: 0.6451 - val_loss: 1.0518 - val_accuracy: 0.6215 - lr: 8.0000e-06
Epoch 79/80
448/448 [==============================] - 33s 74ms/step - loss: 0.9805 - accuracy: 0.6480 - val_loss: 1.0522 - val_accuracy: 0.6217 - lr: 8.0000e-06
Epoch 80/80
448/448 [==============================] - 35s 77ms/step - loss: 0.9803 - accuracy: 0.6438 - val_loss: 1.0526 - val_accuracy: 0.6223 - lr: 8.0000e-06

Results

[13]
import matplotlib.pyplot as plt

fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

[20]
train_loss, train_accu = fernet.evaluate(train_generator)
test_loss, test_accu = fernet.evaluate(test_generator)
print("final train accuracy = {:.2f} , Test accuracy = {:.2f}".format(train_accu*100, test_accu*100))
449/449 [==============================] - 26s 58ms/step - loss: 0.8347 - accuracy: 0.7090
113/113 [==============================] - 3s 29ms/step - loss: 1.0516 - accuracy: 0.6227
final train accuracy = 70.90 , Test accuracy = 62.27

[19]
import numpy as np
from sklearn.metrics import classification_report

# Get predicted labels for test data
y_pred = fernet.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Get true labels for test data
y_true = test_generator.classes

# Get classification report
target_names = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

113/113 [==============================] - 3s 26ms/step
              precision    recall  f1-score   support

       angry       0.51      0.56      0.53       958
     disgust       0.82      0.25      0.39       111
        fear       0.51      0.34      0.41      1024
       happy       0.84      0.84      0.84      1774
     neutral       0.53      0.65      0.58      1233
         sad       0.48      0.50      0.49      1247
    surprise       0.76      0.79      0.77       831

    accuracy                           0.62      7178
   macro avg       0.64      0.56      0.57      7178
weighted avg       0.62      0.62      0.62      7178


Save The Model files

[21]
model_json = fernet.to_json()
with open("/content/drive/MyDrive/emotions/fernet_model2.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
fernet.save_weights('/content/drive/MyDrive/emotions/fernet_model2.h5')
