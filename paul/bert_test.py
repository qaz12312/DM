# coding: utf-8
"""
It is a simple script for test different model.
"""
import random
import pandas as pd
from simpletransformers.classification import ClassificationArgs
from simpletransformers.classification import ClassificationModel


def main() -> None:
    """ Entry point """
    train_data = []
    val_data = []
    test_data = []

    with open("data.log", "r",encoding='utf-8') as f:
        data = f.read().splitlines()

    random.shuffle(data)

    for i in range(len(data)):
        label, feature = data[i].split(" | ")
        label = int(label)

        if i < 700:
            train_data.append([feature, label])
        elif i < 800:
            val_data.append([feature, label])
        else:
            test_data.append(feature)

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_df.columns = ["text", "labels"]
    val_df.columns = ["text", "labels"]
    test_df.columns = ["text"]

    # Args
    model_args = ClassificationArgs(
        overwrite_output_dir=True,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
    )

    # Create a classification model
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args,
    )

    model.train_model(train_df)

    # Predict
    predictions, raw_outputs = model.predict(["今天天氣真好"])
    print(predictions,raw_outputs)


if __name__ == "__main__":
    main()