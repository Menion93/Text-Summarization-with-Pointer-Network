from os import listdir
import string
import numpy as np


def mmap(fn, elem):
    return list(map(fn, elem))

def getel(n, lst): 
    return mmap(lambda x: x[n], lst)
    
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 0]
    return cleaned


# load stories
directory = 'cnn/stories/'
stories = load_stories(directory)
print('Loaded Stories %d' % len(stories))

# clean stories
for example in stories:
    example['story'] = clean_lines(example['story'].split('\n'))
    example['highlights'] = clean_lines(example['highlights'])


# save to file
from pickle import dump, load
dump(stories, open('cnn_dataset.pkl', 'wb'))


# load from file
from pickle import dump, load
stories = load(open('cnn_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))

size = len(stories) // 10
# Set Seed
np.random.seed(seed=13367)
stories = np.array(stories)
indexes = np.random.choice(list(range(stories.shape[0])), size=size, replace=False)

# Sample 10% of dataset
random_sample = stories[indexes]

size = len(stories) // 10
# Set Seed
np.random.seed(seed=13367)
stories = np.array(stories)
indexes = np.random.choice(list(range(92579)), size=size, replace=False)

# Sample 10% of dataset
random_sample = stories[indexes]

def get_first_n_sentence(data, n):
    tmp = data.tolist()
    X = [". ".join(obj['story'][:n]) for obj in tmp]
    y = [". ".join(obj['highlights']) for obj in tmp]
    return X, y

X, y = get_first_n_sentence(random_sample, 10)

indexes = mmap(lambda x:len(x.split(' ')) < 500, X)
X = np.array(X)[indexes]
y = np.array(y)[indexes]

# tokenize with TreebankWord tokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

X_tokens = [tokenizer.tokenize(x) for x in X]
y_tokens = [tokenizer.tokenize(x) for x in y]

from collections import Counter

def flatmap(lst):
    return [word for sent in lst for word in sent]

voc_counts = Counter(flatmap(X_tokens) + flatmap(y_tokens))
# make the cut
voc = voc_counts.most_common()[:13128]

# BUILD THE VOCABULARY

words = getel(0, voc)
w2id = dict(zip(words, range(len(words))))

# add <start>, <end>, <unk> token to vocabulary
w2id['<start>'] = len(w2id.keys())
w2id['<end>'] = len(w2id.keys())
w2id['<unk>'] = len(w2id.keys())
w2id['<pad>'] = len(w2id.keys())

# BUILD THE REVERSE VOCABULARY

id2w = dict(mmap(lambda x: (x[1], x[0]), w2id.items()))

# CONVERT X TOKENS TO INDEXES

X_id = [[w2id.get(word, w2id['<unk>']) for word in sent] for sent in X_tokens]

# ADD START AND END TOKEN AND CONVERT Y TO ID
y_id = [[w2id['<start>']] + \
        [w2id.get(word, w2id['<unk>']) for word in sent] + \
        [w2id['<end>']] for sent in y_tokens]

# SEQUENCE PADDING

from keras.preprocessing.sequence import pad_sequences
X_id=pad_sequences(X_id, value=w2id['<pad>'], padding='post')
y_id = pad_sequences(y_id, value=w2id['<pad>'], padding='post')

# GENERATE Y AND GEN INPUTS
y_id_new = []
pad_index = w2id['<pad>']
def p_gen_index(elem, inp, i, log=False, padding=pad_index):
    if i % 1000 == 0 and log:
        print('starting example n°', i)
    if elem == padding:
        y_id_new.append(padding)
        return padding
    try:
        index = inp.tolist().index(elem)
        y_id_new.append(index)
        return 0
    except:
        y_id_new.append(elem)
        return 1

gen = [[p_gen_index(elem, ex, i, log=i2 == 0)  for i2, elem in enumerate(y)] for i, (ex, y) in enumerate(zip(X_id, y_id))]

y_final = np.array(y_id_new).reshape(y_id.shape)

### TRAIN, VAL, TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_val_test, y_train, \
y_val_test, y_raw_train, y_raw_val_test,\
gen_train, gen_val_test = train_test_split(X_id, y_final, y_id, np.array(gen), test_size=0.20 )

X_val, X_test, y_val, \
y_test, y_raw_val, y_raw_test,\
gen_val, gen_test = train_test_split(X_val_test, y_val_test, y_raw_val_test, gen_val_test, test_size=0.50 )

import os
os.makedirs('processed', exist_ok=True)

np.save('processed/X_train', X_train)
np.save('processed/X_val', X_val)
np.save('processed/X_test', X_test)

np.save('processed/y_train', y_train)
np.save('processed/y_val', y_val)
np.save('processed/y_test', y_test)

np.save('processed/y_raw_train', y_raw_train)
np.save('processed/y_raw_val', y_raw_val)
np.save('processed/y_raw_test', y_raw_test)

np.save('processed/gen_train', gen_train)
np.save('processed/gen_val', gen_val)
np.save('processed/gen_test', gen_test)

# SAVE VOCABULARIES
with open('processed/w2id.pkl', 'wb') as f:
    dump(w2id, f)
    
with open('processed/id2w.pkl', 'wb') as f:
    dump(id2w, f)
