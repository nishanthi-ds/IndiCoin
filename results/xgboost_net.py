import numpy as np
import pandas as pd
import os, re, cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import Network_Input_pipeline
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow as tf

# Train/Test Split
X_train, X_test, y_train, y_test = Network_Input_pipeline.get_data(images_path='./Detected_mendley/',
                                                                   preprocess=preprocess_input,target_path="label2.csv", use_header=False )

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    #preprocessing_function=add_blur,
    validation_split=0.2
)

# 2. Convert labels to categorical
# y_train_cat = to_categorical(y_train, num_classes=num_classes)
# y_test_cat = to_categorical(y_test, num_classes=num_classes)

# 3. Create generators
train_generator = datagen.flow(
    X_train, y_train,
    batch_size=32,
    subset='training'
)
# Step 1: Define CNN model (you already did this)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze convolutional layers

model = Sequential([
    base_model,
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.4),
    Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Step 2: Train the model
model.fit(
    X_train,y_train,
    validation_data=(X_test, y_test),
    epochs= 20,
    # callbacks=[checkpoint]  # Uncomment if using checkpoints
)

model.save('mobilenetoldnow.h5')
# # Step 3: Use MobileNetV2 + GlobalAveragePooling to extract features
# from tensorflow.keras.layers import GlobalAveragePooling2D

# feature_model = Sequential([
#     base_model,
#     GlobalAveragePooling2D()  # Output shape: (None, 1280)
# ])
#
# # Get features for both train and test images
# X_train_features = feature_model.predict(X_train, batch_size=32, verbose=1)
# X_test_features = feature_model.predict(X_test, batch_size=32, verbose=1)
#
# print("Train Features Shape:", X_train_features.shape)
# print("Test Features Shape:", X_test_features.shape)
#
#
# # Train XGBoost
# xgb = XGBClassifier(n_estimators=500, max_depth=50,num_boost_round=100, learning_rate=0.05, use_label_encoder=True, eval_metric='mlogloss')
# # xgb.fit(X_train_features, y_train)
#
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# # Base model: weak learner
# base_model = xgb #DecisionTreeClassifier(max_depth=2)
#
# # AdaBoost setup
# ada_model = AdaBoostClassifier(
#     estimator=base_model,
#     n_estimators=500,           # Increase if needed
#     learning_rate=0.05,          # Tune with grid search
#     algorithm='SAMME',        # Recommended for multiclass with probabilities
#     random_state=42
# )
#
# # Fit the model
# ada_model.fit(X_train_features, y_train)
#
# # Predict
# y_pred = ada_model.predict(X_test_features)
#
# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
