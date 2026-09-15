import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET = "vision/gru_dataset"

SEQUENCE_LENGTH = 30

X = []
y = []


# =====================================
# Load Dataset
# =====================================

for class_name in os.listdir(DATASET):

    class_path = os.path.join(
        DATASET,
        class_name
    )

    if not os.path.isdir(class_path):
        continue


    for filename in os.listdir(class_path):

        if not filename.endswith(".npy"):
            continue


        path = os.path.join(
            class_path,
            filename
        )


        data = np.load(path)


        # ต้องมี 30 frames
        if data.shape[0] < SEQUENCE_LENGTH:
            continue


        # เอา 30 frames แรก
        data = data[:SEQUENCE_LENGTH]


        # (30,17,2)
        # -> (30,34)

        data = data.reshape(
            SEQUENCE_LENGTH,
            -1
        )


        X.append(data)
        y.append(class_name)


X = np.array(X)

y = np.array(y)


print("X:", X.shape)
print("y:", y.shape)


# =====================================
# Encode labels
# =====================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)


print(
    "Classes:",
    encoder.classes_
)


# =====================================
# Train / Validation
# =====================================

X_train, X_val, y_train, y_val = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# =====================================
# GRU Model
# =====================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(30, 34)
    ),

    tf.keras.layers.GRU(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.GRU(
        64
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])


model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# =====================================
# Train
# =====================================

model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=16
)


# =====================================
# Save
# =====================================

os.makedirs(
    "models",
    exist_ok=True
)


model.save(
    "models/gru_action.keras"
)


np.save(
    "models/gru_classes.npy",
    encoder.classes_
)


print("GRU saved!")