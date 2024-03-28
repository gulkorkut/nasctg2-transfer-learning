import pandas as pd
import numpy as np
import io
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

df = pd.read_parquet('data.parquet')

features = df.drop('label', axis=1)
target = df['label']

X_train_first_500, _, y_train_first_500, _ = train_test_split(features, target, train_size=5000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_train_first_500, y_train_first_500, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array

X_train_processed = np.array([preprocess_image(image['bytes']) for image in X_train['image']])
X_val_processed = np.array([preprocess_image(image['bytes']) for image in X_val['image']])
X_test_processed = np.array([preprocess_image(image['bytes']) for image in X_test['image']])


model = Sequential([
  DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3)), 
  GlobalAveragePooling2D(),
  Dense(256, activation='relu'),  
  Dense(128, activation='relu'),  
  Dense(64, activation='relu'),
  Dense(10, activation='softmax')
])


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)


model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['accuracy'])

history = model.fit(X_train_processed, y_train, epochs=10, validation_data=(X_val_processed, y_val))

test_loss, test_acc = model.evaluate(X_test_processed, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

from sklearn.metrics import precision_score, f1_score, recall_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


y_pred = model.predict(X_test_processed)

precision = precision_score(y_test, np.argmax(y_pred, axis=1), average='weighted')
recall = recall_score(y_test, np.argmax(y_pred, axis=1), average='weighted')
f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='weighted')
accuracy = (y_test == np.argmax(y_pred, axis=1)).mean()


rmse = np.sqrt(mean_squared_error(y_test, np.argmax(y_pred, axis=1)))
mae = mean_absolute_error(y_test, np.argmax(y_pred, axis=1))
mape = mean_absolute_percentage_error(y_test, np.argmax(y_pred, axis=1))


print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')

print(df.info())
print(df.head())

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, np.argmax(y_pred, axis=1))

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
print(df.info())
print(df.head())