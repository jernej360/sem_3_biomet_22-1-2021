import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing import image

print(tf.__version__)

import pathlib

DATA_PATH="./ears/"

batch_size = 32
img_height = 80
img_width = 80

def configure_for_performance(ds):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def main(retrain=False):
    data_dir = pathlib.Path(DATA_PATH)
    image_count = len(list(data_dir.glob('*/*.png')))
    print("there are: "+str(image_count)+" photos")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print(class_names)


    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 2

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        #loss='binary_crossentropy',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    if retrain:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10
        )
        model.save("model7_10epoch.h5")
    else:
        model = tf.keras.models.load_model('model6_1000epoch.h5')

    correct=0
    incorrect=0

    for file in os.listdir("./ears_test"):
        img_pred = image.load_img("./ears_test/" + file, target_size=(img_width, img_height))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)
        rslt = model.predict(img_pred)

        category = file.split('.')[0]
        if category == 'm' and np.argmax(rslt) == 1:
            correct=correct+1
        elif category == 'f' and np.argmax(rslt) == 0:
            correct=correct+1
        else:
            incorrect=incorrect+1

        print("./ears_test/" + file + "   " + str(rslt)+" --------    "+str(np.argmax(rslt)))

    print("finish results are:   correct "+str(correct)+" /  incorect "+str(incorrect))



if __name__ == "__main__":
    main()
