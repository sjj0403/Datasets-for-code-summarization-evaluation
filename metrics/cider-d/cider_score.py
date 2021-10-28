import copy
from collections import defaultdict
import numpy as np
import pdb
import math
import pandas as pd 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string
stop_words = stopwords.words('english')
print(stop_words)
punc = string.punctuation.split()
print(punc)
print(type(punc))
def wordnetSim(word1, word2):
    # phrasel1 = word1
    # phrasel2 = word2
    # word1 = phrasel1.split(' ')
    # word2 = phrasel2.split(' ')
    path_sim = 0
    for w1 in word1:
        for w2 in word2:
            synsets1 = wn.synsets(w1)
            synsets2 = wn.synsets(w2)
            for tmpword1 in synsets1:
                for tmpword2 in synsets2:
                    if tmpword1.pos() == tmpword2.pos():
                        try:
                            sim = tmpword1.path_similarity(tmpword2)
                            if w1 != w2:
                                path_sim = max(path_sim, sim)
                        except Exception as e:
                            continue
    return path_sim

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(float)
    # print(type(s))
    # print(s)
    
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    # print(counts)
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''
    Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''
    Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n, True)

class CiderScorer(object):
    """
    CIDEr scorer.
    """

    def copy(self):
        ''' copy the refs.'''
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None
        path = '/media/shenjuanjuan/新加卷1/comment evaluation/data process/dataset/sjj/select/select.pkl'
        source = pd.read_pickle(path)
        # self.md = source['methodname_st'].tolist()
        # self.api = source['api_st'].tolist()
        # self.iden = source['iden_st'].tolist()
        
    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test)) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def update_docpre(self):
        for (ngram, doc_freq) in self.document_frequency.items():
            count = 0
            for word in ngram:
                if word in stop_words or word in punc:
                    self.document_frequency[ngram] *= 2

    def update_stopws(self):
        for i in range(len(stop_words)):
            word = tuple(stop_words[i:i+1])
            if self.document_frequency[word] > 0:
                self.document_frequency[word] = 8000
        for i in range(len(punc)):
            pc = tuple(punc[i:i+1])
            if self.document_frequency[pc] > 0:
                self.document_frequency[pc] = 8000
    
    def update_refn(self):
        index = 0
        for (s, refs) in zip(self.md, self.crefs):
            l = str(s).split()
            for ngram in refs[0].items():
                count = 0
                for word in ngram:
                    if word in stop_words:
                        count += 1
                    elif word in punc:
                        count += 1                        
                    elif word in l:
                        # self.crefs[index][0][ngram] *= 1.1
                        count += 1
                if count == len(ngram):
                    self.crefs[index][0][ngram] *= 1.2
            index += 1

    def update_tesn(self):
        index = 0
        for s in self.md:
            l = str(s).split()
            for ngram in self.ctest[index]:
                count = 0
                for word in ngram:
                    if word in stop_words :
                        count += 1
                    elif word in punc:
                        count += 1
                    elif word in l:
                        # self.ctest[index][ngram] *= 1.1
                        count += 1
                if count == len(ngram):
                    self.ctest[index][ngram] *= 1.2
                        
            index += 1

    def update_ref(self):
        index = 0
        for s in self.md:
            l = str(s).split()
            for i in range(len(l)):
                ngram = tuple(l[i:i+1])
                if self.crefs[index][0][ngram] > 0:
                    #print(self.crefs[index][0][ngram])
                    self.crefs[index][0][ngram] *= np.log(6318)/(np.log(6318)-np.log(self.document_frequency[ngram]))/math.exp(1)
                    #print(self.crefs[index][0][ngram])
            index += 1
        
        index = 0
        for s in self.api:
            
            l = str(s).split()
            for i in range(len(l)):
                ngram = tuple(l[i:i+1])
                if self.crefs[index][0][ngram] > 0:
                    self.crefs[index][0][ngram] *= np.log(6318)/(np.log(6318)-np.log(self.document_frequency[ngram]))/math.exp(1)
            index += 1
        index = 0
        for s in self.iden:
            l = str(s).split()
            for i in range(len(l)):
                ngram = tuple(l[i:i+1])
                if self.crefs[index][0][ngram] > 0:
                    self.crefs[index][0][ngram] *= 1#np.log(6319)/(np.log(6319)-np.log(self.document_frequency[ngram]))/math.exp(1)
            index += 1

    def update_tes(self):
        index = 0
        for s in self.md:
            l = str(s).split()
            for i in range(len(l)):
                ngram = tuple(l[i:i+1])
                if self.ctest[index][ngram] > 0 and self.document_frequency[ngram] > 0:
                    self.ctest[index][ngram] *= np.log(6318)/(np.log(6318)-np.log(self.document_frequency[ngram]))/math.exp(1)
            index += 1
        
        index = 0
        for s in self.api:
            l = str(s).split()
            for i in range(len(l)):
                ngram = tuple(l[i:i+1])
                if self.ctest[index][ngram] > 0 and self.document_frequency[ngram] > 0:
                    self.ctest[index][ngram] *= np.log(6318)/(np.log(6318)-np.log(self.document_frequency[ngram]))/math.exp(1)
            index += 1
        index = 0
        for s in self.iden:
            l = str(s).split()
            for i in range(len(l)):
                ngram = tuple(l[i:i+1])
                if self.ctest[index][ngram] > 0 and self.document_frequency[ngram] > 0:
                    self.ctest[index][ngram] *= 1#np.log(6319)/(np.log(6319)-np.log(self.document_frequency[ngram]))/math.exp(1)
            index += 1    
    #     for ref in self.crefs:

    #         for j in range(len(words)):
    #     ngram = tuple(words[i:i+1])
    #     counts[ngram] += 1
    # for j in range(len(words)):
    #     ngram = tuple(words[i:i+1])
    #     if str(ngram) in mn:
    #         flag = 3
    #     elif str(ngram) in api:
    #         flag = 2
    #     elif str(ngram) in ide:
    #         flag = 2
    #     counts[ngram] *= flag


    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                #print(self.ref_len)#8.751316246773456
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length


        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # delta = float(min(0, length_ref-length_hyp))
            # delta = 0
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                    # val[n] += vec_hyp[n][ngram]* vec_ref[n][ngram]
                    

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])
               
                

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val


        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        if len(self.crefs) == 1:
            self.ref_len = 1
        scores = []
        
        for test, refs in zip(self.ctest, self.crefs):  
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # print(vec)
            # print(norm)
            # print(length)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # score_avg = 3/4*np.mean(score) + 1/4*score[0]
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)

        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        # self.update_docpre()
        # self.update_stopws()
        # # self.update_refn()
        # # self.update_tesn()
        # self.update_ref()
        # self.update_tes()
        
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)