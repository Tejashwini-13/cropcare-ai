import tensorflow as tf
import json
import os
import numpy as np

from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset"

IMG_SIZE = (128, 128)

BATCH_SIZE = 16

EPOCHS = 10

MODEL_PATH = "model/multi_crop_model.keras"

CLASS_NAMES_PATH = "model/class_names.json"


# ==========================================
# CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...\n")


train_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    validation_split=0.2,

    subset="training",

    seed=123

)


validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False,

    validation_split=0.2,

    subset="validation",

    seed=123

)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print("\n===================================")
print("CLASSES FOUND")
print("===================================")

print("Number of classes:", len(class_names))

for i, name in enumerate(class_names):

    print(i, ":", name)


# ==========================================
# SAVE CLASS NAMES
# ==========================================

with open(CLASS_NAMES_PATH, "w") as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ==========================================
# CHECK DATASET
# ==========================================

print("\nChecking dataset...\n")


class_counts = np.zeros(
    len(class_names),
    dtype=np.int32
)


for images, labels in train_dataset:

    labels_numpy = labels.numpy()

    for label in labels_numpy:

        class_counts[label] += 1


print("Images per class:")

for i, count in enumerate(class_counts):

    print(
        class_names[i],
        ":",
        count
    )


# ==========================================
# CLASS WEIGHTS
# ==========================================

class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.arange(len(class_names)),

    y=np.repeat(
        np.arange(len(class_names)),
        class_counts
    )

)


class_weights = {

    i: float(class_weights_array[i])

    for i in range(len(class_names))

}


print("\nClass weights created.")


# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_dataset.prefetch(
    AUTOTUNE
)


validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),

])


# ==========================================
# BASE MODEL
# ==========================================

print("\nLoading MobileNetV2...\n")


base_model = tf.keras.applications.MobileNetV2(

    input_shape=(128, 128, 3),

    include_top=False,

    weights="imagenet"

)


# Freeze base model initially

base_model.trainable = False


# ==========================================
# MODEL
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    data_augmentation,

    tf.keras.layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.4
    ),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=3,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True

    )

]


# ==========================================
# TRAIN
# ==========================================

print("\n===================================")
print("STARTING TRAINING")
print("===================================\n")


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    MODEL_PATH
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print(
    "Model saved:",
    MODEL_PATH
)

print(
    "Classes saved:",
    CLASS_NAMES_PATH
)

print(
    "Number of classes:",
    len(class_names)
)

print("\nCropCare AI model is ready!")