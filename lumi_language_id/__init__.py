import contextlib
import os
import unicodedata
from math import log2
from pathlib import Path

import fasttext
import numpy as np

import ftfy
import langcodes


FT_LANGUAGES = [
    'af', 'als', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'av', 'az', 'azb', 'ba',
    'bar', 'bcl', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'bxr', 'ca',
    'cbk', 'ce', 'ceb', 'ckb', 'co', 'cs', 'cv', 'cy', 'da', 'de', 'diq',
    'dsb', 'dty', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi',
    'fr', 'frr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gom', 'gu', 'gv', 'he', 'hi',
    'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ie', 'ilo', 'io', 'is',
    'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv',
    'kw', 'ky', 'la', 'lb', 'lez', 'li', 'lmo', 'lo', 'lrc', 'lt', 'lv', 'mai',
    'mg', 'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl', 'my',
    'myv', 'mzn', 'nah', 'nap', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'oc',
    'or', 'os', 'pa', 'pam', 'pfl', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm',
    'ro', 'ru', 'rue', 'sa', 'sah', 'sc', 'scn', 'sco', 'sd', 'sh', 'si', 'sk',
    'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk',
    'tl', 'tr', 'tt', 'tyv', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls',
    'vo', 'wa', 'war', 'wuu', 'xal', 'xmf', 'yi', 'yo', 'yue', 'zh'
]


CACHED_MODEL = None

def detect_language(text):
    """
    Get the detected language of a string, and its confidence.

    This loads a TunedLanguageIdentifier (imported late to avoid circular
    imports) the first time it's run, and stores it in a global variable.
    This can be managed more explicitly using `TunedLanguageIdentifier.load()`
    in `lumi_language_id.tuned`, but it helps to provide a convenient
    top-level function here.
    """
    global CACHED_MODEL
    from lumi_language_id.tuned import TunedLanguageIdentifier

    if CACHED_MODEL is None:
        CACHED_MODEL = TunedLanguageIdentifier.load()

    return CACHED_MODEL.detect_language(text)


def corpus_file(path):
    """
    Get a data file from a standard location. Check a few possible locations until the file is
    found.
    """
    my_location = Path(__file__).parent
    root_location = my_location.parent
    nfs_location = Path('/nfs/mg-cache/language_id/corpus')
    paths_to_try = [root_location / 'corpus', nfs_location]
    for location in paths_to_try:
        path_to_try = location / path
        if path_to_try.exists():
            return str(path_to_try)

    raise FileNotFoundError(f"Can't find {path!r} in any of {paths_to_try!r}")


def data_file(path):
    """
    Get a path to a file in the local 'data' directory.
    """
    my_location = Path(__file__).parent
    return str(my_location / 'data' / path)


def align_language_to_fasttext(language):
    """
    Given a language code, get the closest-matching language code that fastText would detect,
    or 'und' if there is no match.

    >>> align_language_to_fasttext('fr')
    'fr'
    >>> align_language_to_fasttext('iw')  # old language code for Hebrew
    'he'
    >>> align_language_to_fasttext('zz')  # not a language
    'und'
    >>> align_language_to_fasttext('nan')
    'zh'
    >>> align_language_to_fasttext('nb')
    'no'
    """
    matched_language, _dist = langcodes.closest_match(language, FT_LANGUAGES)
    return matched_language


def predicted_info(confidence):
    """
    Convert the estimated confidence of a language ID prediction into a number of bits of
    surprisal if the prediction is wrong, with a maximum of 20.

    Examples:

    >>> predicted_info(0.5)
    1.
    >>> predicted_info(0.875)
    3.
    >>> predicted_info(1.)
    20.
    """
    if confidence >= 1.:
        return 20.
    return min(20., -log2(1. - confidence))


def clean_text(text):
    """
    Clean text for better language detection, keeping only letters, 'marks', and whitespace.
    (Marks are used in some languages for diacritical marks that appear over letters.)
    """
    cleaned_text = ftfy.fix_text(text, normalization='NFKC').casefold()

    # Keep only letters (L), marks (M), and whitespace (Z)
    kept_chars = [ch for ch in cleaned_text if unicodedata.category(ch)[0] in 'LMZ']
    cleaned_text = ''.join(kept_chars)

    # Remove extra whitespaces
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text


class LanguageIdentifier:
    def __init__(self, name='lid.176.ftz'):
        """
        Create a LanguageIdentifier object that stores a loaded language-ID model and uses it
        to identify languages.

        The optional 'name' is the filename of the model that should be looked for in this
        module's standard paths.
        """
        # Open a FastText model without sending a blank line to stdout
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.ft_model = fasttext.load_model(data_file(name))

    def detect_language(self, text):
        """
        Predict the language of a text using fastText.

        Returns a pair of the detected language code and its confidence (from 0 to 1).
        """
        cleaned = clean_text(text)
        prediction_struct = self.ft_model.predict(cleaned)

        # The structure returned by fastText looks like:
        # (('__label__en',), array([0.99047953]))

        language_struct, confidence_struct = prediction_struct

        # Extract the predicted language code from this:
        label_size = len('__label__')
        pred_language = language_struct[0][label_size:]

        # In the wonderful future of Python 3.9, the above would be:
        # language_struct[0].removeprefix('__label__')

        # And then the confidence in the prediction:
        pred_confidence = confidence_struct[0]
        return (pred_language, pred_confidence)

    def make_data_point(self, text):
        """
        Given a piece of text, convert it to an input and output that can be used to train
        a 'tuned' classifier that re-estimates the confidence of a prediction.
        """
        text = clean_text(text)
        num_spaces = text.count(' ')
        text_length = len(text)
        num_han_characters = 0
        for char in text:
            if 0x3300 <= ord(char) < 0xa000:
                num_han_characters += 1
        language, confidence = self.detect_language(text)
        info = predicted_info(confidence)

        return np.array([text_length, info, num_spaces, num_han_characters]), language
