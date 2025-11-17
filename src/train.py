# src/train.py

from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import joblib
import os

def train_lg(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    os.makedirs("models", exist_ok=True)  # Create the folder if not already there
    joblib.dump(model, "models/lg_model.pkl")


def train_nn(x_train, y_train):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.PReLU(), #learns slope itself

        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.PReLU(),

        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.LeakyReLU(negative_slope=0.01)
        tf.keras.layers.PReLU(),
        # tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        # tf.keras.layers.LeakyReLU(negative_slope=0.01),
        # tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
        # tf.keras.layers.LeakyReLU(negative_slope=0.01),
        # tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.PReLU(),
        # tf.keras.layers.LeakyReLU(negative_slope=0.01),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),

        # tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(4, activation='softmax'),
    ])
    nn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(x_train, y_train, epochs=300, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stop])

    os.makedirs("models", exist_ok=True)
    nn_model.save("models/nn_model.h5")