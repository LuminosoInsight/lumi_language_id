from setuptools import setup

setup(
    name="lumi_language_id",
    version='0.2.1',
    maintainer='Robyn Speer',
    maintainer_email='rspeer@luminoso.com',
    description='For when you want fastText language identification, but you also want to believe the answers',
    license='MIT',
    platforms=["any"],
    packages=['lumi_language_id'],
    package_data={'lumi_language_id': ['data/*.npz', 'data/*.ftz']},
    install_requires=['fasttext', 'numpy', 'ftfy', 'langcodes >= 2'],

    # Training the tuned model requires scikit-learn, but using the model doesn't.
    extras_require={
        'train': ['scikit-learn'],
    },
    python_requires='>=3.6',
)
