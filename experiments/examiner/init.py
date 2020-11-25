import spacy
import csv
import numpy as np
import tqdm

def examiner():
    # https://www.kaggle.com/therohk/examine-the-examiner/
    data = csv.DictReader(open("data/examiner-date-text.csv", "r"), delimiter=",", dialect="excel")
    nlp = spacy.load("en_core_web_lg")
    dataset = []
    dates, texts = zip(*[(x["publish_date"], x["headline_text"]) for x in data])
    for date, text in tqdm.tqdm(zip(dates, nlp.pipe(texts, batch_size=50, disable=["tagger", "parser", "ner"])), total=3089781):
        dataset.append(text.vector)
        # print(date, text.vector)
        # break
    dataset = np.array(dataset)
    print(dataset.shape)
    np.save("data/examiner.npy", dataset)

if __name__ == "__main__":
    examiner()
