import numpy      as np
import pandas     as pd
import tensorflow as tf
import math       as mt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

train_dir    = '/Users/gusta/OneDrive/Área de Trabalho/MODELS/ResNet_transfer_learning/0.Code/dogs-vs-cats/train'
val_dir      = '/Users/gusta/OneDrive/Área de Trabalho/MODELS/ResNet_transfer_learning/0.Code/dogs-vs-cats/validation'
test_dir     = '/Users/gusta/OneDrive/Área de Trabalho/MODELS/ResNet_transfer_learning/0.Code/dogs-vs-cats/test'

num_classes  = 2
img_width    = 224
img_height   = 224
batch_number = 64

train_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                     rotation_range = 20,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.2
                                    )
valid_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
test_generator  = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_generator.flow_from_directory(train_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size  = batch_number,
                                                      shuffle     = True,
                                                      seed        = 666,
                                                      class_mode  = 'categorical'                                                   
                                                     )
validation_generator = valid_generator.flow_from_directory(val_dir,
                                                           target_size = (img_width, img_height),
                                                           batch_size  = batch_number,
                                                           shuffle     = False,
                                                           seed        = 666,
                                                           class_mode  = 'categorical'                                                   
                                                          )
test_generator = valid_generator.flow_from_directory(test_dir,
                                                     target_size = (img_width, img_height),
                                                     batch_size  = batch_number,
                                                     shuffle     = False,
                                                     seed        = 666,
                                                     class_mode  = 'categorical'                                                   
                                                    )

def modelo():
    modelo_base      = ResNet50(weights='imagenet', 
                                include_top = False, 
                                input_shape = (img_width, img_height, 3)                               
                               )
    for layer in modelo_base.layers[:]:
        layer.trainable = False
    
    input = Input(shape = (img_width, img_height, 3))
    modelo_customizado  = modelo_base(input)
    modelo_customizado  = GlobalAveragePooling2D()(modelo_customizado)
    modelo_customizado  = Dense(64,
                                activation = 'relu')(modelo_customizado)
    modelo_customizado  = Dropout(0.5)(modelo_customizado)
    predictions         = Dense(num_classes,
                                activation = 'softmax')(modelo_customizado)
    return Model(inputs = input, outputs = predictions)

path, dirs, files = next(os.walk(os.path.join(train_dir, 'cats').replace("\\","/")))
count_cats_train = len([x for x in files])
path, dirs, files = next(os.walk(os.path.join(train_dir, 'dogs').replace("\\","/")))
count_dogs_train = len([x for x in files])

path, dirs, files = next(os.walk(os.path.join(val_dir, 'cats').replace("\\","/")))
count_cats_val = len([x for x in files])
path, dirs, files = next(os.walk(os.path.join(val_dir, 'dogs').replace("\\","/")))
count_dogs_val = len([x for x in files])

model = modelo()
model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.optimizers.Adam(lr = 0.001),
              metrics = ['acc', tf.keras.metrics.AUC()])
#steps_train = mt.ceil(float(count_cats_train + count_dogs_train)/batch_number)
#steps_valid = mt.ceil(float(count_cats_val + count_dogs_val)/batch_number)
steps_train = mt.ceil(float(500)/batch_number)
model.fit_generator(train_generator,
                    steps_per_epoch  = steps_train,
                    epochs           = 6,
                    validation_data  = validation_generator,
                    validation_steps = steps_train
                   )

model.save('model.h5')

