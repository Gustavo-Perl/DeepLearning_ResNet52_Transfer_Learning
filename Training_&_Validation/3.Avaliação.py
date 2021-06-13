import numpy             as np
import pandas            as pd
import tensorflow        as tf
import math              as mt
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models              import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics                      import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats                          import ks_2samp

train_dir    = '/Users/gusta/OneDrive/Área de Trabalho/MODELS/ResNet_transfer_learning/0.Code/dogs-vs-cats/train'
val_dir      = '/Users/gusta/OneDrive/Área de Trabalho/MODELS/ResNet_transfer_learning/0.Code/dogs-vs-cats/validation'
test_dir     = '/Users/gusta/OneDrive/Área de Trabalho/MODELS/ResNet_transfer_learning/0.Code/dogs-vs-cats/test'

num_classes  = 2
img_width    = 224
img_height   = 224
batch_number = 64

train_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
valid_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
test_generator  = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_generator.flow_from_directory(train_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size  = batch_number,
                                                      shuffle     = False,
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

model = load_model('model.h5')

groud_truth_train = train_generator.classes #0 é cat, 1 é dog
train_pred        = model.predict(train_generator)

groud_truth_valid = validation_generator.classes
valid_pred        = model.predict(validation_generator)

groud_truth_test  = test_generator.classes
test_pred         = model.predict(test_generator)

groud_truth_train = pd.DataFrame(groud_truth_train).rename(columns = {0: 'Real'})
groud_truth_train['Max_Proba'] = pd.DataFrame(train_pred)[1]
groud_truth_train['Max_Index'] = pd.DataFrame(train_pred).idxmax(axis=1)

groud_truth_valid = pd.DataFrame(groud_truth_valid).rename(columns = {0: 'Real'})
groud_truth_valid['Max_Proba'] = pd.DataFrame(valid_pred)[1]
groud_truth_valid['Max_Index'] = pd.DataFrame(valid_pred).idxmax(axis=1)

groud_truth_test = pd.DataFrame(groud_truth_test).rename(columns = {0: 'Real'})
groud_truth_test['Max_Proba'] = pd.DataFrame(test_pred)[1]
groud_truth_test['Max_Index'] = pd.DataFrame(test_pred).idxmax(axis=1)

COMPARATIVO = pd.DataFrame(columns = {'BASE', 'KS', 'ROC', 'ACC', 'PREC', 'RECALL', 'F1'})
COMPARATIVO.at[0, 'BASE']   = 'Treino'
COMPARATIVO.at[0, 'KS']     = round(ks_2samp(np.asarray(groud_truth_train['Real']), 
                                             np.asarray(groud_truth_train['Max_Proba']))[0]*100, 2)
COMPARATIVO.at[0, 'ROC']    = round(roc_auc_score(np.asarray(groud_truth_train['Real']), 
                                                  np.asarray(groud_truth_train['Max_Proba']))*100, 2)
COMPARATIVO.at[0, 'ACC']    = round(accuracy_score(np.asarray(groud_truth_train['Real']), 
                                                   np.asarray(groud_truth_train['Max_Index']))*100, 2)
COMPARATIVO.at[0, 'PREC']   = round(precision_score(np.asarray(groud_truth_train['Real']), 
                                                    np.asarray(groud_truth_train['Max_Index']))*100, 2)
COMPARATIVO.at[0, 'RECALL'] = round(recall_score(np.asarray(groud_truth_train['Real']), 
                                                 np.asarray(groud_truth_train['Max_Index']))*100, 2)
COMPARATIVO.at[0, 'F1']     = round(f1_score(np.asarray(groud_truth_train['Real']), 
                                             np.asarray(groud_truth_train['Max_Index']))*100, 2)

COMPARATIVO.at[1, 'BASE']   = 'Valid'
COMPARATIVO.at[1, 'KS']     = round(ks_2samp(np.asarray(groud_truth_valid['Real']), 
                                             np.asarray(groud_truth_valid['Max_Proba']))[0]*100, 2)
COMPARATIVO.at[1, 'ROC']    = round(roc_auc_score(np.asarray(groud_truth_valid['Real']), 
                                                  np.asarray(groud_truth_valid['Max_Proba']))*100, 2)
COMPARATIVO.at[1, 'ACC']    = round(accuracy_score(np.asarray(groud_truth_valid['Real']), 
                                                   np.asarray(groud_truth_valid['Max_Index']))*100, 2)
COMPARATIVO.at[1, 'PREC']   = round(precision_score(np.asarray(groud_truth_valid['Real']), 
                                                    np.asarray(groud_truth_valid['Max_Index']))*100, 2)
COMPARATIVO.at[1, 'RECALL'] = round(recall_score(np.asarray(groud_truth_valid['Real']), 
                                                 np.asarray(groud_truth_valid['Max_Index']))*100, 2)
COMPARATIVO.at[1, 'F1']     = round(f1_score(np.asarray(groud_truth_valid['Real']), 
                                             np.asarray(groud_truth_valid['Max_Index']))*100, 2)

COMPARATIVO.at[2, 'BASE']   = 'Test'
COMPARATIVO.at[2, 'KS']     = round(ks_2samp(np.asarray(groud_truth_test['Real']), 
                                             np.asarray(groud_truth_test['Max_Proba']))[0]*100, 2)
COMPARATIVO.at[2, 'ROC']    = round(roc_auc_score(np.asarray(groud_truth_test['Real']), 
                                                  np.asarray(groud_truth_test['Max_Proba']))*100, 2)
COMPARATIVO.at[2, 'ACC']    = round(accuracy_score(np.asarray(groud_truth_test['Real']), 
                                                   np.asarray(groud_truth_test['Max_Index']))*100, 2)
COMPARATIVO.at[2, 'PREC']   = round(precision_score(np.asarray(groud_truth_test['Real']), 
                                                    np.asarray(groud_truth_test['Max_Index']))*100, 2)
COMPARATIVO.at[2, 'RECALL'] = round(recall_score(np.asarray(groud_truth_test['Real']), 
                                                 np.asarray(groud_truth_test['Max_Index']))*100, 2)
COMPARATIVO.at[2, 'F1']     = round(f1_score(np.asarray(groud_truth_test['Real']), 
                                             np.asarray(groud_truth_test['Max_Index']))*100, 2)
COMPARATIVO = COMPARATIVO[['BASE', 'KS', 'ROC', 'ACC', 'PREC', 'RECALL', 'F1']]
COMPARATIVO

title_font = {'fontname' : 'Arial',
              'size'     : '17',
              'weight'   : 'bold'}
axis_font  = {'fontname' : 'Arial',
              'size'     : '12'}
lr_fpr_valid_train, lr_tpr_valid_train, _ = roc_curve(np.asarray(groud_truth_train['Real']), 
                                                      np.asarray(groud_truth_train['Max_Proba']))
lr_fpr_valid_valid, lr_tpr_valid_valid, _ = roc_curve(np.asarray(groud_truth_valid['Real']), 
                                                      np.asarray(groud_truth_valid['Max_Proba']))
lr_fpr_valid_test, lr_tpr_valid_test, _   = roc_curve(np.asarray(groud_truth_test['Real']), 
                                                      np.asarray(groud_truth_test['Max_Proba']))

plt.plot([0.0, 1.0], [0.0, 1.0], 'r--', linewidth = 0.5, label = 'Coin', color = 'black')
plt.plot(lr_fpr_valid_train,
         lr_tpr_valid_train,
         linewidth = 0.5, 
         label = 'Treino - ' + str(round(roc_auc_score(np.asarray(groud_truth_train['Real']), 
                                                       np.asarray(groud_truth_train['Max_Proba']))*100, 2)) + ' %', 
         color = 'red')
plt.plot(lr_fpr_valid_valid,
         lr_tpr_valid_valid,
         linewidth = 0.5, 
         label = 'Valid - ' + str(round(roc_auc_score(np.asarray(groud_truth_valid['Real']), 
                                                       np.asarray(groud_truth_valid['Max_Proba']))*100, 2)) + ' %', 
         color = 'blue')
plt.plot(lr_fpr_valid_test,
         lr_tpr_valid_test,
         linewidth = 0.5, 
         label = 'Test - ' + str(round(roc_auc_score(np.asarray(groud_truth_test['Real']), 
                                                       np.asarray(groud_truth_test['Max_Proba']))*100, 2)) + ' %', 
         color = 'darkgreen')
plt.title('ROC Curves', title_font)
plt.xlabel('False Positive Rate', axis_font)
plt.ylabel('True Positive Rate', axis_font)
plt.legend()
plt.show()