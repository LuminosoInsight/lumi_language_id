from setuptools import setup

setup(
    name="lumi_language_id",
    version='0.1',
    maintainer='Robyn Speer',
    maintainer_email='rspeer@luminoso.com',
    platforms=["any"],
    packages=['lumi_language_id'],
    install_requires=['fasttext'],
    python_requires='>=3.5',
)
