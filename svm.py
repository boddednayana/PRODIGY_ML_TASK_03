import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
cat_folder=r"C:\Users\konat\OneDrive\PRODIGY_ML_TASK_03\PRODIGY_ML_TASK_03\dogs_cats_sample_1000\train\cats"
dog_folder=r"C:\Users\konat\OneDrive\PRODIGY_ML_TASK_03\PRODIGY_ML_TASK_03\dogs_cats_sample_1000\train\dogs"
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:  # Check if the image was loaded successfully
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img)
            labels.append(label)
        else:
            print(f"Failed to load image: {img_path}")  # Debugging statement
    return images, labels

cats, cat_labels = load_images(cat_folder, 0)  # Label cats as 0
dogs, dog_labels = load_images(dog_folder, 1)  # Label dogs as 1

# Combine data
X = np.array(cats + dogs)
y = np.array(cat_labels + dog_labels)

# Print shapes for debugging
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Normalize data
X = X.astype('float32') / 255.0
X = X.reshape(X.shape[0], -1)  # Flatten the images

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear')  # You can experiment with different kernels
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, target_names=['Cats', 'Dogs']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Cats', 'Dogs'])
plt.yticks(tick_marks, ['Cats', 'Dogs'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()