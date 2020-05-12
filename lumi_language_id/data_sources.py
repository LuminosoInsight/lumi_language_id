"""
Functions that read training/test data from corpus files.
"""
import csv
import ftfy

from lumi_language_id import corpus_file, align_language_to_fasttext


def twitter_gen():
    with open(
        corpus_file('TweetLID_corpusV2/tweetlid-test-tweets.tsv')
    ) as twitter_file:
        reader = csv.reader(twitter_file, delimiter='\t')
        for row in reader:
            # If there are multiple possibilities separated by +, take the first one
            label = row[2].split('+')[0]
            text = ''.join(x for x in row[3] if x.isprintable()).replace('\n', ' ')
            text = ftfy.fix_text(text)
            fixed_label = align_language_to_fasttext(label)
            if fixed_label != 'und':
                yield (text, fixed_label)


def wiki_gen():
    with open(corpus_file('WiLI/x_test.txt')) as texts_file:
        with open(corpus_file('WiLI/y_test.txt')) as labels_file:
            for text, label in zip(texts_file, labels_file):
                label = label.rstrip()
                text = ''.join(x for x in text if x.isprintable()).replace('\n', ' ')
                text = ftfy.fix_text(text)
                fixed_label = align_language_to_fasttext(label)
                if fixed_label != 'und':
                    yield (text, fixed_label)


def tatoeba_gen():
    with open(corpus_file('Tatoeba/tatoeba_short_text.txt')) as tatoeba_file:
        for line in tatoeba_file:
            line = line.split('\t')
            text = line[1].replace('\n', ' ')
            text = ftfy.fix_text(text)
            label = line[0]
            fixed_label = align_language_to_fasttext(label)
            if fixed_label != 'und':
                yield (text, fixed_label)
