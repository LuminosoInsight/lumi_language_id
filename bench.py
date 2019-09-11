import fasttext
import argparse
import json
import csv
from tabulate import tabulate
from tqdm import tqdm
import pycld2 as cld2
from iso639 import languages

parser = argparse.ArgumentParser()
# parser.add_argument('--twitter', action='store_true')
# parser.add_argument('--wikipedia', action='store_true')

args = parser.parse_args()

ft_small = fasttext.load_model('lid.176.ftz')
ft_large = fasttext.load_model('lid.176.bin')

def twitter_gen():
    with open('./data/TweetLID_corpusV2/tweetlid-test-tweets.tsv') as twitter_file:
        reader = csv.reader(twitter_file, delimiter='\t')
        for row in reader:
            label = row[2]
            text = row[3].replace('\n', ' ')
            yield (text, label)

def wiki_gen():
    wiki2label = {}
    with open('./data/WiLI/labels.csv') as label_file:
        reader = csv.reader(label_file, delimiter=';')
        next(reader)
        for row in reader:
            try:
                wiki2label[row[3]] = languages.get(part3=row[3]).part1
            except:
                continue
    with open('./data/WiLI/x_test.txt') as texts_file:
        with open('./data/WiLI/y_test.txt') as labels_file:
            for text, label in zip(texts_file, labels_file):
                label = label.rstrip()
                if not label in wiki2label or wiki2label[label] == '':
                    continue

                text = ''.join(x for x in text if x.isprintable())
                for t in text.split('\n'):
                    yield (t, wiki2label[label])

def evaluate(gen):
    total = 0
    sm_score = 0
    lg_score = 0
    cld2_score = 0
    for text, label  in gen:
        total += 1

        sm_pred = ft_small.predict(text)
        lg_pred = ft_large.predict(text)
        cld2_pred = cld2.detect(text, isPlainText=True, bestEffort=True)

        sm_pred = sm_pred[0][0][-2:]
        lg_pred = lg_pred[0][0][-2:]
        cld2_pred = cld2_pred[2][0][1][:2]

        if sm_pred in label:
            sm_score += 1
        if lg_pred in label:
            lg_score += 1
        if cld2_pred in label:
            cld2_score += 1
    print(tabulate([['fastText Small', round(sm_score / total, 2)],
                    ['fastText Large', round(lg_score / total, 2)],
                    ['CLD2', round(cld2_score / total, 2)]],
                    headers=['Model', 'Acc']))
print("Twitter")
evaluate(twitter_gen())
print()

print("Wikipedia")
evaluate(wiki_gen())
print()
