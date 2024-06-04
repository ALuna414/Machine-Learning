#************************************************************************************
# Aaron Luna
# ML – HW#5
# Filename: DCNN.py
# Due: Nov. 1, 2023
#
# Objective:
# -Develop your own CNN model to classify all classes.
# -Provide the training and test confusion matrices.
# -Provide the test accuracy, precision, recall, and F1-scores to a text file.
# -Provide the Loss and Accuracy curves for training and validation (you can use a single plot for these four curves)
#************************************************************************************
#Importing all required libraries
print('\nImporting Packages........')
import numpy as np
import pandas as pd
import os
import shutil
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
print('\t\t\t\t........DONE!')


####################################################
#
# Data Preprocessing
#
####################################################
print('\nData Preprocessing........')
# Path set for each of the folders we want to extract from
#original_dataset_dir = '/Documents/HW5/CCIC/'

base_dir = '/gpfs/home/a_l523/HW5/CCIC/Negative'
base_dir2 = '/gpfs/home/a_l523/HW5/CCIC/Positive'

train_dir = '/gpfs/home/a_l523/HW5/Dataset/Train'
validation_dir = '/gpfs/home/a_l523/HW5/Dataset/Val'
test_dir = '/gpfs/home/a_l523/HW5/Dataset/Test'

test_neg_dir = '/gpfs/home/a_l523/HW5/Dataset/Test/Neg'
test_pos_dir = '/gpfs/home/a_l523/HW5/Dataset/Test/Pos'

val_neg_dir = '/gpfs/home/a_l523/HW5/Dataset/Val/Neg'
val_pos_dir = '/gpfs/home/a_l523/HW5/Dataset/Val/Pos'

train_neg_dir = '/gpfs/home/a_l523/HW5/Dataset/Train/Neg'
train_pos_dir = '/gpfs/home/a_l523/HW5/Dataset/Train/Pos'

train_confusion_matrix = '/gpfs/home/a_l523/HW5/train_confusion_matrix.png'
test_confusion_matrix = '/gpfs/home/a_l523/HW5/test_confusion_matrix.png'
txt_location = '/gpfs/home/a_l523/HW5/metrics.txt'
loss_acc_plot = '/gpfs/home/a_l523/HW5/loss_acc_plot.png'

i = 0
for filename in os.listdir(base_dir):
   if i < 1000: 
      shutil.copyfile(base_dir + filename, train_neg_dir + '/' + filename)
   elif i < 1500: 
      shutil.copyfile(base_dir + filename, val_neg_dir + '/' + filename)
   elif i < 2000: 
      shutil.copyfile(base_dir + filename, test_neg_dir + '/' + filename)
   else:
      break
   i += 1

i = 0
for filename in os.listdir(base_dir2):
   if i < 1000:
      shutil.copyfile(base_dir2 + filename, train_pos_dir + '/' + filename)
   elif i < 1500:
      shutil.copyfile(base_dir2 + filename, val_pos_dir + '/' + filename)
   elif i < 2000:
      shutil.copyfile(base_dir2 + filename, test_pos_dir + '/' + filename)
   else:
      break
   i += 1
print('\t\t\t\t........DONE!')
####################################################
# End of Data Preprocessing CODE
####################################################


####################################################
#
# Metrics Code
#
####################################################
print('\nMetrics Processing........')
def write_metrics(y_test, y_test_pred, y_train, y_train_pred, txt_location):
   # Calculate and print the accuracy
   test_accuracy = accuracy_score(y_test, y_test_pred)
   print("test Accuracy: %.2f%%" % (test_accuracy * 100.0))
   train_accuracy = accuracy_score(y_train, y_train_pred)
   print("train Accuracy: %.2f%%" % (train_accuracy * 100.0))
   print("\n\n\n")
   classification_report_str = classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive'])
   print(classification_report_str)
   with open(txt_location, 'w') as file:
      file.write("test Accuracy: %.2f%%" % (test_accuracy * 100.0))
      file.write("train Accuracy: %.2f%%" % (train_accuracy * 100.0))
      file.write("\n\n\n")
      file.write(classification_report_str)
print('\t\t\t\t........DONE!')
####################################################
# End of metrics CODE
####################################################


####################################################
#
# Model Creation Code
#
####################################################
print('\nModel Creating........')
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(227, 227), batch_size=32, class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(227, 227), batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(227, 227), batch_size=32, class_mode='binary')

model = models.Sequential()
model.add(layers.Conv2D(32,(3, 3),activation='relu',input_shape=(227,227,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])
history = model.fit(train_generator,
                    steps_per_epoch=40, #number of gradient step before next epoch
                    epochs=2,
                    validation_data=validation_generator,
                    validation_steps=20)
test_loss, test_acc = model.evaluate(test_images, test_labels)
model.save('posneg_cracks.h5')
print('\t\t\t\t........DONE!')
####################################################
# End of model creation CODE
####################################################


####################################################
#
# Matrix Train and Test outcome code
#
####################################################
print('\nMatrix Making........')
def plot_confusion_matrix(y_labels, y_preds, save_location, title):
   print("Plotting " + title + " confusion matrix...")
   confusion_data = confusion_matrix(y_labels, y_preds)
   plt.figure(figsize=(8, 6))
   sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16})
   plt.xlabel("Predicted")
   plt.ylabel("True")
   plt.title(title)
   plt.savefig(save_location)
   #plt.show()
   plt.close()
print('\t\t\t\t........DONE!')
####################################################
# End of matrix train CODE
####################################################


####################################################
#
# Best Results code
#
####################################################
print('\nBest Results........')
# Predictions for best model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n\n\nTrain Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}\n")
print(f"Recall: {recall:.2f}\n")
print(f"F1: {f1:.2f}\n")
print("\n\n\n")

# print to a txt file
with open('CNN_results.txt', 'w') as file:
    file.write(f"Best Train Accuracy: {train_accuracy:.2f}")
    file.write(f"\nBest Test Accuracy: {accuracy:.2f}")
    file.write(f"\nPrecision Score: {precision:.2f}")
    file.write(f"\nRecall Score: {recall:.2f}")
    file.write(f"\nF1 Score: {f1:.2f}\n")
print('\t\t\t\t........DONE!')
####################################################
# End of best results CODE
####################################################


####################################################
#
# Loss and Accuracy code
#
####################################################
print('\nLoss and Accuracy........')
def plot_loss_acc(history, save_location):
   print("Plotting loss and accuracy...")
   acc = history.history['acc']
   val_acc = history.history['val_acc']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   epochs = range(1, len(acc) + 1)
   # Create subplots
   plt.figure(figsize=(12, 6))

   # Subplot for accuracy
   plt.subplot(1, 2, 1)
   plt.plot(epochs, acc, 'bo', label='Training acc')
   plt.plot(epochs, val_acc, 'b', label='Validation acc')
   plt.title('Training and validation accuracy')
   plt.legend()

   # Subplot for loss
   plt.subplot(1, 2, 2)
   plt.plot(epochs, loss, 'bo', label='Training loss')
   plt.plot(epochs, val_loss, 'b', label='Validation loss')
   plt.title('Training and validation loss')
   plt.legend()

   plt.savefig(save_location)
   plt.close()
print('\t\t\t\t........DONE!')
####################################################
# End of loss and accuracy CODE
####################################################


####################################################
#
# Predictions code
#
####################################################
print('\nPredictions........')
train_generator.reset()
test_generator.reset()
validation_generator.reset()

y_train = train_generator.classes
y_test = test_generator.classes

train_predictions = model.predict(train_generator, steps=len(train_generator), verbose=1)
y_train_pred = [1 * (x[0]>=0.5) for x in train_predictions]

test_predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_test_pred = [1 * (x[0]>=0.5) for x in test_predictions]
print('\t\t\t\t........DONE!')
####################################################
# End of predictions CODE
#################################################### 


####################################################
#
# Function calls
#
####################################################
plot_confusion_matrix(y_train, y_train_pred, train_confusion_matrix, 'Train Confusion Matrix')
plot_confusion_matrix(y_test, y_test_pred, test_confusion_matrix, 'Test Confusion Matrix')

plot_loss_acc(history, loss_acc_plot)

write_metrics(y_test, y_test_pred, y_train, y_train_pred, txt_location)
####################################################
# End of function calls
####################################################