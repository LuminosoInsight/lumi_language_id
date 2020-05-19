import itertools
import langcodes
import numpy as np

from lumi_language_id import LanguageIdentifier, data_file
from lumi_language_id.data_sources import twitter_gen, wiki_gen, tatoeba_gen
from lumi_language_id.tuned import MultiLayerPerceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, log_loss


def get_training_data():
    """
    The data used for training and validation comes from a labeled Twitter corpus and
    a corpus of Wikipedia article introductions.
    """
    return make_input_and_output(itertools.chain(twitter_gen(), wiki_gen()))


def get_test_data():
    """
    The held-out test data comes from Tatoeba. Sentences in Tatoeba tend to be
    'pedagogical' examples that are short examples of correct, standardized
    writing in each language -- this means they would be unrepresentative to
    train on, but make for reliable test data.
    """
    return make_input_and_output(tatoeba_gen())


def make_input_and_output(input_gen):
    """
    Convert a generator of labeled examples into the 'X' and 'y' arrays that
    are used to train scikit-learn.
    """
    inputs = []
    outputs = []
    labels = []
    lid = LanguageIdentifier()
    for text, label in input_gen:
        row, detected_lang = lid.make_data_point(text)
        match = langcodes.tag_distance(label, detected_lang) <= 5
        inputs.append(row)
        outputs.append(match)
        labels.append(label)
    return np.array(inputs), np.array(outputs), labels


def make_estimator():
    """
    Make a classifier that estimates the probability of the underlying langID
    model's classification.

    The form of the classifier is a multi-layer perceptron, which I settled on
    after trying multiple different models. An advantage of an MLP is that it
    is trained on log-loss, which rewards correctly calibrated probabilities.

    There are only three input features, so we wouldn't get much advantage from
    a hidden layer size whose dimensionality is much higher than that -- but I
    found that having 8 hidden features is advantageous. My intuition here is
    that ReLU features must output a constant 0 below (or above) their
    intercept, while a sum of two ReLU features has the ability to vary in the
    positive or negative direction, providing another degree of freedom.

    Two hidden layers learned this function better than one, and I saw no
    advantage from having 3 or 4 hidden layers.

    You'd think that using a model-selection tool like AutoML would have been
    the answer here, but AutoML told me to use a random forest classifier,
    which is absolutely wrong when the goal is to output well-tuned
    probabilities.
    """
    return MLPClassifier(
        activation='relu', hidden_layer_sizes=(8, 8), alpha=0.1, max_iter=1000
    )


def run():
    """
    Build the tuned model that re-estimates the probability of a language
    classification.
    """
    estimator = make_estimator()
    input_train, output_train, _labels = get_training_data()

    # Split our two 'training' sources into training and validation. Though
    # we don't have a separate validation step -- it's treated the same as
    # test data -- the purpose here is to show the performance of the classifier
    # on held-out data that is like the training data, while the actual
    # test data is from an entirely held-out data set (Tatoeba).
    input_train, input_val, output_train, output_val = train_test_split(
        input_train, output_train, test_size=0.2
    )

    # Train the estimator
    estimator.fit(input_train, output_train)

    # Determine the log loss of using fastText's original "confidence" value as-is,
    # comparing it against the validation labels. This number is too high.
    orig_confidence = input_val[:, 1]
    orig_loss = log_loss(output_val, orig_confidence)
    print(f'Original log loss: {orig_loss:3.3f}')

    # Use the trained estimator to make new predictions on the validation data.
    predictions = estimator.predict(input_val)
    predictions_p = estimator.predict_proba(input_val)[:, 1]

    # Show the accuracy and log-loss of the new estimator.
    model_accuracy = balanced_accuracy_score(predictions, output_val)
    model_loss = log_loss(output_val, predictions_p)
    print(f'Validation accuracy: {model_accuracy:3.3f}')
    print(f'Log loss: {model_loss:3.3f}')

    # Get the same statistics for the test data.
    input_test, output_test, labels_test = get_test_data()
    predictions = estimator.predict(input_test)
    predictions_p = estimator.predict_proba(input_test)[:, 1]
    model_accuracy = balanced_accuracy_score(predictions, output_test)
    model_loss = log_loss(output_test, predictions_p)
    print(f'Test accuracy: {model_accuracy:3.3f}')
    print(f'Log loss: {model_loss:3.3f}')

    # Show a breakdown of test set accuracy per language
    languages = sorted(set(labels_test))
    for lang in languages:
        lang_filter = [label == lang for label in labels_test]
        lang_accuracy = balanced_accuracy_score(predictions[lang_filter], output_test[lang_filter])
        print(f'\t{lang}\t{lang_accuracy:3.3f}')

    # Extract the trained model and save it in a form that doesn't require
    # scikit-learn to run.
    coefs = estimator.coefs_
    intercepts = estimator.intercepts_
    MultiLayerPerceptron.save(data_file('tuned.npz'), coefs, intercepts)


if __name__ == '__main__':
    run()
