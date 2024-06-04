# Import necessary libraries
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from matplotlib import pyplot as plt


# Step 1: Load and Pre-process Data
def load_data(dataset_path):
    data = []
    labels = []

    # Loop through each file in the dataset directory
    for file in os.listdir(dataset_path):
        # Load the audio file using 
      item_path = os.path.join(dataset_path, file)
      if os.path.isfile(item_path):
        signal, sr = librosa.load(os.path.join(dataset_path, file))
        
        # Extract MFCCs and Chroma (additional feature for harmonic analysis)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        
        # Combine and standardize features
        combined_features = np.vstack((mfccs, chroma))
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(combined_features)

        # Append data and labels
        data.append(standardized_features)
        labels.append(file.split('_')[0])
    
    return np.array(data), labels

data, labels = load_data('/mmfs1/home/a_l523/Project/Dataset/IRMAS-TrainingData/voi')

# Convert labels to one-hot encoded format
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Standardize features
scaler = StandardScaler()
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten the features
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Reshape back to 3D if necessary (depends on your model's input shape)
# Assuming the LSTM expects a 3D input (samples, timesteps, features)
timesteps = 1  # Change this according to your sequence length
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], timesteps, -1)
X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], timesteps, -1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], timesteps, -1)


# Step 2: Define the Model
model = Sequential()

# LSTM layer
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.25))

#model.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile with Adam optimizer and learning rate scheduler
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callbacks for model saving and early stopping
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Step 3: Train the Model
model.fit(X_train, y_train, 
          batch_size=32, epochs=50, 
          validation_data=(X_val, y_val), 
          callbacks=[checkpoint, early_stop])

# Step 4: Evaluate the Model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Step 5: Make predictions
y_pred = model.predict(X_test)

# Convert one-hot encoded labels back to categorical labels
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_labels)
print("Confusion Matrix:\n", conf_matrix)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
#Matplotlib’s matshow
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):
   for j in range(confmat.shape[1]):
      ax.text(x=j, y=i,
      s=confmat[i, j],
      va='center', ha='center')
plt.xlabel('predicted  test label')
plt.ylabel('true test label')
plt.savefig('ConfusionMatrix.png')
plt.show() 
plt.close()
