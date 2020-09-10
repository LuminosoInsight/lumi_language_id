from lumi_language_id.tuned import TunedLanguageIdentifier


LID = TunedLanguageIdentifier.load()


def _get_language(text):
    lang, prob = LID.detect_language(text)
    return lang


def test_language_id():
    assert _get_language("here's some words") == 'en'
    assert _get_language("aquí hay algunas palabras") == 'es'
    assert _get_language("dette er ord som passer godt sammen") == 'nb'
    assert _get_language("これらは言葉です") == 'ja'
    assert _get_language("这些是单词") == 'zh'
