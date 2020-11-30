#/bin/bash

kaggle datasets download -d therohk/million-headlines
mkdir data
unzip million-headlines.zip -d data
rm million-headlines.zip
python -m spacy download en_core_web_lg
python init.py
