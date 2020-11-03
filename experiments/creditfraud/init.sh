#/bin/bash

kaggle datasets download mlg-ulb/creditcardfraud
mkdir data
unzip creditcardfraud.zip -d data
rm creditcardfraud.zip