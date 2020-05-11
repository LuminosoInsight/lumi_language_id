from setuptools import setup

setup(
    name="lumi_language_id",
    version='0.1',
    maintainer='Robyn Speer',
    maintainer_email='rspeer@luminoso.com',
    platforms=["any"],
    packages=['lumi_language_id'],
    install_requires=['fasttext', 'numpy', 'ftfy', 'langcodes >= 2'],

    # Training the tuned model requires scikit-learn, but using the model doesn't.
    extras_require={
        'train': ['scikit-learn'],
    }
    python_requires='>=3.5',
)
