import numpy as np
from lumi_language_id import LanguageIdentifier, data_file


class MultiLayerPerceptron:
    """
    A simple implementation of an MLP classifier. This implementation has no training code, but
    can be used to run a 'frozen' classifier that was trained by scikit-learn.
    """

    def __init__(self, coefs, intercepts):
        self.coefs = coefs
        self.intercepts = intercepts

    def forward(self, row):
        """
        Propagate a set of input features (`row`) through the MLP classifier, and get the output
        features.

        We assume here that all layers but the last one are ReLU layers, and the last one is
        a sigmoid layer.
        """
        for layer_num, (coefs_layer, intercepts_layer) in enumerate(
            zip(self.coefs, self.intercepts)
        ):
            row = row @ coefs_layer + intercepts_layer
            if layer_num < len(self.coefs) - 1:
                # Apply ReLU activation
                row = np.maximum(row, 0.)
            else:
                # Apply sigmoid activation
                row = 1. / (np.exp(-row) + 1.)

        return row

    def probability(self, row):
        """
        Return the probability that this row of input belongs to the 'True' class, in a
        binary classifier.
        """
        # Binary classifiers are represented with one output in the final layer. Extract this
        # output from its array
        return self.forward(row)[0]

    @staticmethod
    def save(filename, coefs, intercepts):
        """
        Save the coefficients and intercepts of a trained classifier in a .npz
        file that can be loaded without scikit-learn.
        """
        version = 1
        n_layers = len(coefs)
        arrays = {'meta': np.array([version, n_layers])}
        for layer_num, (coefs_layer, intercepts_layer) in enumerate(
            zip(coefs, intercepts)
        ):
            arrays[f'coefs_{layer_num}'] = coefs_layer
            arrays[f'intercepts_{layer_num}'] = intercepts_layer
        np.savez(filename, **arrays)

    @classmethod
    def load(cls, filename):
        """
        Load a MultiLayerPerceptron classifier from a .npz file.
        """
        arrays = np.load(filename)
        version, n_layers = arrays['meta']
        if version != 1:
            raise NotImplementedError(
                "This code only understands MultiLayerPerceptron version 1"
            )

        coefs = []
        intercepts = []
        for layer_num in range(n_layers):
            coefs_layer = arrays[f'coefs_{layer_num}']
            intercepts_layer = arrays[f'intercepts_{layer_num}']
            coefs.append(coefs_layer)
            intercepts.append(intercepts_layer)

        return cls(coefs, intercepts)


class TunedLanguageIdentifier:
    """
    A FastText language ID classifier with another classifier on top of it, so
    it can produce reliable probability estimates. It will refrain from
    detecting a language if the probability of the detection being correct is
    less than 0.5.
    """

    def __init__(
        self,
        language_identifier: LanguageIdentifier,
        tuned_classifier: MultiLayerPerceptron,
    ):
        self.language_identifier = language_identifier
        self.tuned_classifier = tuned_classifier

    @classmethod
    def load(cls, fasttext_filename='lid.176.ftz', tuned_filename='tuned.npz'):
        """
        Load a TunedLanguageIdentifier from its fastText file and an .npz file
        of the retuned classifier.
        
        The filenames refer to files in the `lumi_language_id/data` directory,
        and default to a classifier that's included with the repository, so
        `TunedLanguageIdentifier.load()` should get you a classifier.
        """
        lid = LanguageIdentifier(fasttext_filename)
        tuned = MultiLayerPerceptron.load(data_file(tuned_filename))
        return cls(lid, tuned)

    def detect_language(self, text):
        """
        Predict the language of a text using fastText.

        Returns a pair of the detected language code and its probability (from
        0.5 to 1).  If the probability we detect is less than 0.5, the detected
        language becomes 'und', so that we're not returning an answer that's
        probably wrong.
        """
        row, language = self.language_identifier.make_data_point(text)
        probability = self.tuned_classifier.probability(row)
        if probability < 0.5:
            language = 'und'

        return language, probability
