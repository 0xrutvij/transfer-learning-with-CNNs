import argparse
import os
import sys

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

# Keras outputs warnings using `print` to stderr so let's direct that to devnull temporarily
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator

# we're done
sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PathLike = Union[str, Path]

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

CLASSES = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy',
           'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']


def validate_file(file_path: str):
    fp = Path(file_path)
    condition1 = fp.exists() and fp.is_file()
    # Supported image formats: jpeg, png, bmp, gif. Animated gifs are truncated to the first frame.
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    condition2 = fp.suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return condition1 and condition2


def dataframe_from_csv(csv_file_path: PathLike):
    df = pd.read_csv(csv_file_path, skipinitialspace=True)
    cols = set(df.columns)

    if "image_path" not in cols or "label" not in cols:
        print("Error: Invalid CSV File, Header Should be of form [image_path, label].")
        exit()

    paths_exist = df["image_path"].apply(validate_file)

    if not paths_exist.all():
        print("Error: The images at following paths aren't valid or don't exist. \n")
        print(df[~paths_exist]["image_path"].to_string())
        exit()

    return df


def test_data_from_dataframe(df: pd.DataFrame):
    data_generator = ImageDataGenerator()
    return data_generator.flow_from_dataframe(
        dataframe=df,
        x_col="image_path",
        y_col="label",
        class_mode="sparse",
        classes=CLASSES,
        target_size=IMG_SIZE,
        batch_size=32,
        shuffle=False
    )


def load_model_at_path(mpath: PathLike):
    model = keras.models.load_model(mpath)
    model.summary()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Test")

    parser.add_argument("-m", '--model', type=str, default='trained_models/inception_inaturalist_flowers',
                        help='Saved model')

    parser.add_argument("-csv", '--test_csv', type=str, default='flowers_test.csv',
                        help='CSV file with true labels, header must be {image_path, label}')

    parser.add_argument("-spl", "--show-prediction-labels", action="store_true",
                        help="Show a table of actual and predicted labels")

    args = parser.parse_args()
    model_path = args.model
    test_csv_path = args.test_csv
    show_prediction_labels = args.show_prediction_labels

    test_df = dataframe_from_csv(test_csv_path)
    test_data = test_data_from_dataframe(test_df)
    print("\n\n")

    model_to_test = load_model_at_path(model_path)
    print("\n\n")

    if show_prediction_labels:
        raw_predictions = model_to_test.predict(test_data)
        predictions = np.argmax(raw_predictions, axis=1)
        test_df["prediction"] = [CLASSES[i] for i in predictions]
        correct_predictions = len(test_df[test_df["label"] == test_df["prediction"]])
        total = len(test_df)
        acc = correct_predictions / total
        print(test_df.to_string())
    else:
        _, acc = model_to_test.evaluate(test_data, verbose=2)

    print("\n\n")
    print("-" * 49)
    print('|\tTest model, accuracy: {:5.5f}%         |'.format(100 * acc))
    print("-" * 49)
    