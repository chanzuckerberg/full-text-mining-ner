from setuptools import setup
import os
import sys

setup(
    name='meta_full_text_mining_ner',
    version='1.0.0',
    packages=['meta_full_text_mining_ner'],
    package_dir={'meta_full_text_mining_ner': '.'},
    package_data={'meta_full_text_mining_ner': ['data']},
    url='https://github.com/chanzuckerberg/meta-full-text-mining-ner',
    license='MIT ',
    author='Ana-Maria Istrate',
    author_email='aistrate@chanzuckerberg.com',
    description='Methods and dataset named entity recognition (NER) for full-text biomedical research articles.',
    install_requires=[
        'pandas==1.1.2', 'nltk==3.5', 'torch==1.6.0', 'transformers==3.1.0', 'scikit-learn==0.23.1'
    ]
)