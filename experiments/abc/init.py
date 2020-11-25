import spacy
import csv
import numpy as np
import tqdm

def abc():
    # https://www.kaggle.com/therohk/million-headlines
    data = csv.DictReader(open("data/abcnews-date-text.csv", "r"), delimiter=",", dialect="excel")
    nlp = spacy.load("en_core_web_lg")
    dataset = []
    dates, texts = zip(*[(x["publish_date"], x["headline_text"]) for x in data])
    for date, text in tqdm.tqdm(zip(dates, nlp.pipe(texts, batch_size=50, disable=["tagger", "parser", "ner"])), total=1186018):
        dataset.append(text.vector)
        # print(date, text.vector)
        # break
    dataset = np.array(dataset)
    print(dataset.shape)
    np.save("data/abc.npy", dataset)


if __name__ == "__main__":
    abc()
