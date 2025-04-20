from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import Input_pipeline
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint

def agumentation(X_train,y_train):

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        validation_split=0.1,
    )

    # Create generators
    train_generator = datagen.flow(
        X_train, y_train,
        batch_size=32,
        subset='training'
    )

    return train_generator

class Mobilenet_model():

    def __init__(self,shape, num_class):
        self.INPUT_SHAPE = shape #  = (224, 224, 3)
        self.NUM_CLASS = num_class
        self.OUT_ACTIVATION = 'softmax'

    def get_base_model(self):
        self.base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.INPUT_SHAPE)
        self.base_model.trainable = False  # Freeze convolutional layers
        return self.base_model

    def  get_model(self):

        base_model = self.get_base_model()
        self.model = Sequential([
            base_model,
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.4),
            Dense(self.NUM_CLASS, activation=self.OUT_ACTIVATION )
        ])

        return self.model


if __name__ == '__main__':

    # Train/Test Split
    X_train, X_test, y_train, y_test = Input_pipeline.get_data(images_path='./Detected_mendley/',
                                            preprocess=preprocess_input,
                                            random_state=1,
                                            train_size=0.7,
                                            target_path="label2.csv",
                                            use_header=False )

    # Apply Image Agumentation
    train_generator = agumentation(X_train,y_train)


    # Train model

    mobilenet = Mobilenet_model(shape= (224, 224, 3), num_class=5)

    model = mobilenet.get_model()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',       # File to save the best model
        monitor='val_loss',             # Metric to monitor
        mode='min',                     # 'min' because we want lowest val_loss
        save_best_only=True,            # Only save if the val_loss improves
        verbose=0                       # Print when model is saved
    )

    # Step 2: Train the model
    model.fit(
        train_generator,
        validation_data=(X_test, y_test),
        epochs=30,
        callbacks=[checkpoint],
        batch_size= 32,
        verbose=1
    )
    # model.save('model.h5')

    # Predict
    y_pred=model.predict(X_test)
    y_pred = y_pred.argmax(axis=-1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print(classification_report(y_test, y_pred))
