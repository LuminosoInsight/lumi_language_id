import itertools
import langcodes
import numpy as np

from lumi_language_id import LanguageIdentifier
from lumi_language_id.data_sources import twitter_gen, wiki_gen, tatoeba_gen
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, log_loss


def get_training_data():
    return make_input_and_output(
        itertools.chain(twitter_gen(), wiki_gen())
    )


def get_test_data():
    return make_input_and_output(
        tatoeba_gen()
    )


def make_input_and_output(input_gen):
    inputs = []
    outputs = []
    lid = LanguageIdentifier()
    for text, label in input_gen:
        row, detected_lang = lid.make_data_point(text)
        match = (langcodes.tag_distance(label, detected_lang) <= 5)
        inputs.append(row)
        outputs.append(match)
    return np.array(inputs), np.array(outputs)


def make_estimator():
    return MLPClassifier(activation='relu', hidden_layer_sizes=(6, 6), alpha=0.1, max_iter=1000)
    # return SVC(kernel='linear', probability=True)


def run():
    estimator = make_estimator()
    input_train, output_train = get_training_data()
    input_train, input_val, output_train, output_val = train_test_split(input_train, output_train, test_size=0.2)
    estimator.fit(input_train, output_train)
    input_test, output_test = get_test_data()


    orig_confidence = input_val[:, 1]
    weights_val = np.ones(output_val.shape) * output_val + np.ones(output_val.shape)
    # orig_output = np.array([True] * len(orig_confidence))
    orig_loss = log_loss(output_val, orig_confidence, labels=[False,True], sample_weight=weights_val)
    print(f'Original log loss: {orig_loss:3.3f}')

    predictions = estimator.predict(input_val)
    model_accuracy = balanced_accuracy_score(predictions, output_val)
    
    predictions_p = estimator.predict_proba(input_val)[:,1]
    model_loss = log_loss(output_val, predictions_p, sample_weight=weights_val)
    print(f'Validation accuracy: {model_accuracy:3.3f}')
    print(f'Log loss: {model_loss:3.3f}')

    predictions = estimator.predict(input_test)
    model_accuracy = balanced_accuracy_score(predictions, output_test)
    predictions_p = estimator.predict_proba(input_test)[:,1]
    model_loss = log_loss(output_test, predictions_p)
    print(f'Test accuracy: {model_accuracy:3.3f}')
    print(f'Log loss: {model_loss:3.3f}')

    return estimator


if __name__ == '__main__':
    run()
