import numpy as np
from lumi_language_id import LanguageIdentifier

class MultiLayerPerceptron:
    def __init__(self, coefs, intercepts):
        self.coefs = coefs
        self.intercepts = intercepts

    def forward(self, row):
        for layer_num, (coefs, intercepts) in enumerate(zip(self.coefs, self.intercepts)):
            row = row @ coefs + intercepts
            if layer_num < len(self.coefs) - 1:
                row = np.maximum(row, 0.)
            else:
                row = 1. / (np.exp(-row) + 1.)
        return row

class TunedLanguageIdentifier(LanguageIdentifier):
    pass

