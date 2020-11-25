#/bin/bash

kaggle datasets download -d therohk/examine-the-examiner
mkdir data
unzip examine-the-examiner.zip -d data
rm examine-the-examiner.zip
python -m spacy download en_core_web_lg
python init.py
