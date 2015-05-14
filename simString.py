import pickle
import logging
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
from cleanSent import cleanSent

test_reviewSent_score = []
#pickle.dump (test_reviewSent_score, open ( "test_reviewSent_score.p", "wb") )

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#1. Create a dictionary of raw data without loading all texts into memory: (only has useful words)
dictionary = corpora.Dictionary(line.lower().split() for line in pickle.load(open("tipData.p","rb")))
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
        for line in pickle.load(open("tipData.p","rb")):
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus() # doesn't load the corpus into memory.
corpora.MmCorpus.serialize('corpus.mm', corpus_memory_friendly) # Save corpus to disk
corpus = corpora.MmCorpus('corpus.mm') # Load corpus

#3. Initialize(train/get) the TF-IDF model:
tfidf = models.TfidfModel(corpus)
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

#4. for new string, compute:
sortedSentenceList1 = pickle.load ( open ( "judgerTestSet.p", "rb") )
test_size = 2000
sortedSentenceList = sortedSentenceList1[:test_size]
# for every string in judgeTestSet
for k in range(0, test_size):
    new_sent = cleanSent(sortedSentenceList[k][2])#
    new_vec = dictionary.doc2bow(new_sent) #new_sent.lower().split())
    sims = index[tfidf[new_vec]] # sim array to each node
    #print(sum(sims) / len(sims)) # array
    score = sum(sims)/len(sims)
    test_reviewSent_score.append(score)

outfile = open('simResult', 'w')
outfile.write(str(test_reviewSent_score))
outfile.close()
#print test_reviewSent_score
#pickle.dump (test_reviewSent_score, open ( "test_reviewSent_score.p", "wb") )
#need: reviewid, sentence id? according to judgerTestSet, and score
    # sortedSentenceList1 = pickle.load ( open ( "judgerTestSet.p", "rb") )
    # sortedSentenceList = sortedSentenceList1[:2000]
    # create a list of score representing the score of each sentence(same order). write to file