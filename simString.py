import logging
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import cleanSent

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#1. Create a dictionary of raw data without loading all texts into memory: (only has useful words)
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# remove stop words and words that appear only once
stoplist = set(stopwords.words('english'))
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
dictionary.save('tip.dict') # store the dictionary, for future reference

#2. Convert the training corpus to vector space: (based on raw data, but projected onto dictionary, only useful words)
class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory.
corpora.MmCorpus.serialize('corpus.mm', corpus_memory_friendly) # Save corpus to disk
corpus = corpora.MmCorpus('corpus.mm') # Load corpus

#3. Initialize(train/get) the TF-IDF model:
tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

#4. for new string, compute:
# for every string in judgeTestSet
new_sent = cleanSent("Human computer interaction")#
new_vec = dictionary.doc2bow(new_sent.lower().split())
sims = index[tfidf[new_vec]] # sim array to each node
print(list(enumerate(sims))) # array
