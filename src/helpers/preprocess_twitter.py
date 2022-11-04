"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu (github.com/tokestermw)

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

this version from gist.github.com/ppope > preprocess_twitter.py

light edits by amackcrane, mostly inspired by the test case given at bottom

Edited by André
- Added lambda functions to make code more concise
- Changed test prints
- Moved test prints to main function
- Applied general refactoring

All inline comments come from the original script (mostly by amackcrane)
"""

import argparse
import regex as re
import os

FLAGS = re.MULTILINE | re.DOTALL

def parse_arguments():
    '''Parse the command line arguments.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-p', '--process',
        action='store_true',
        help='set the program into processing mode '
        '(default: testing mode)'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='../../data/original',
        help='the path of the folder containing the data '
        '(default: ../../data/original)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='../../data/twitter_processed',
        help='the path of the folder for the data output '
        '(default: ../../data/twitter_processed)'
    )

    return parser.parse_args()


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result


def tweet_processor(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    
    re_sub = lambda pattern, repl : re.sub(pattern, repl, text, flags=FLAGS)
    allcaps = lambda text : text.group().lower() + " <allcaps> "

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\w+", hashtag)  # edit
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    #text = re_sub(r"([A-Z]){2,}", allcaps)  # moved below

    # additions
    text = re_sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2")
    text = re_sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )")
    text = re_sub(r"  ", r" ")
    text = re_sub(r" ([A-Z]){2,} ", allcaps)
    
    return text.lower()


def testing():
    text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    text2 = "TEStiNg some *tough* #CASES" # couple extra tests
    text_trainset = "@USER She should ask a few native Americans what their take on this is."

    print("Text 1\nOriginal text:\t'{}'\nProcessed text:\t'{}'".format(text, tweet_processor(text)))
    print("")
    print("Text 2\nOriginal text:\t'{}'\nProcessed text:\t'{}'".format(text2, tweet_processor(text2)))
    print("")
    print("Text trainset\nOriginal text:\t'{}'\nProcessed text:\t'{}'".format(text_trainset, tweet_processor(text_trainset)))


def importer(args, tsv):
    folder = args.input

    with open(f'{folder}/{tsv}', 'r', encoding='utf-8') as f:
        data = f.readlines()

    return {i : data[i].split('\t') for i in range(len(data))}


def exporter(args, tsv, tweets_dict):
    folder = args.output

    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(f'{folder}/{tsv}', 'w', encoding='utf-8') as f:
        for i in range(len(tweets_dict)):
            f.write("\t".join(tweets_dict[i]))


def processing(args):
    ''''''
    tsvs = ('train.tsv', 'dev.tsv', 'test.tsv')

    for tsv in tsvs:
        tweets_dict = importer(args, tsv)

        for tweet in range(len(tweets_dict.keys())):
            processed_tweet = tweet_processor(tweets_dict[tweet][0])
            tweets_dict[tweet][0] = processed_tweet

        exporter(args, tsv, tweets_dict)


def main():
    args = parse_arguments()
    
    if args.process:
        processing(args)
    else:
        testing()

if __name__ == '__main__':
    main()
