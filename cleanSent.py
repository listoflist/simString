import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

def cleanSent(strSent):
    stopset = set(stopwords.words('english'))
    stemmer = nltk.PorterStemmer()
    tokens = WordPunctTokenizer().tokenize(strSent)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    final = [stemmer.stem(word) for word in clean]
    return final
