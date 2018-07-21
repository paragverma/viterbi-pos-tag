# -*- coding: utf-8 -*-
import nltk
import pickle
import os
import warnings
import numpy as np
class ViterbiPOSTagger:
    
    def __init__(self, sentstart = 0, sentend = float('inf'), corpus = 'brown', trimtag = 2):
        self.corpus = corpus
        self.customcorpus = None
        self.model = {}
        self.sentstart = sentstart
        self.sentend = sentend
        self.trimtag = trimtag
        self.missin_probs = 1e-6
        self.cpd_tags = None
        self.cpd_tagwords = None
        self.tags_list = None
        
        self.convert_to_list = False
    #Returns Error Message and True/False for validity
    
    """Improve error messages. Pinpoint exact indices"""
    """Perform error checking while training"""
    def check_corpus_validity(self, train_sents):
        if train_sents.__class__.__name__ not in ["list", "ConcatenatedCorpusView", "LazySubsequence"]:
            if self.convert_to_list == True:
                train_sents = list(train_sents)
            else:
                return ("Not a List. Type is: " + train_sents.__class__.__name__, False)
        
        for i in range(len(train_sents)):
            if type(train_sents[i]) is not list:
                return ("Sub list element: " + str(i) + " is not a tuple. " +  str(train_sents[i]) + " is a " + str(type(train_sents[i])), False)
            
            
            
            for j in range(len(train_sents[i])):
                tuple_element = train_sents[i][j]
                if type(tuple_element) is not tuple:
                    return ("List element: " + str(i) + " . " + str(tuple_element) + " is not a tuple.", False)
                if len(tuple_element) != 2:
                    return ("List element: [" + str(i) + "][" + str(j) + "] should be a 2 tuple. Found a " + str(len(train_sents[i])) + " tuple.", False)
                for sub_tuple_element in tuple_element:
                    if type(sub_tuple_element) is not str:
                        return ("Sub tuple element: " + str(i) + " . " + str(tuple_element) + " is not a string.", False)
        return ("Valid Corpus", True)

    def train(self, train_sents):
        cvc = self.check_corpus_validity(train_sents)
        if cvc[1] == False:
            raise TypeError(cvc[0])
            return
        train_tags_words = [ ]
        i = 0
        for sent in train_sents[self.sentstart:min(len(train_sents), self.sentend)]:
            print("Training: {}".format(i))
            i += 1
            # sent is a list of word/tag pairs
            # add START/START at the beginning
            train_tags_words.append( ("START", "START") )
            # then all the tag/word pairs for the word/tag pairs in the sentence.
            # shorten tags to 2 characters each
            train_tags_words.extend([ (tag[:self.trimtag], word) for (word, tag) in sent ])
            # then END/END
            train_tags_words.append( ("END", "END") )
        
        # conditional frequency distribution
        cfd_tagwords = nltk.ConditionalFreqDist(train_tags_words)
        # conditional probability distribution
        """cpd_tagwords contains emission probabilities"""
        self.cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
        
        """print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
        print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))
        """
        
        # Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
        # P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
        train_tags = [tag for (tag, word) in train_tags_words ]
        self.tags_list = list(set(train_tags))
        
        # make conditional frequency distribution:
        # count(t{i-1} ti)
        cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(train_tags))
        # make conditional probability distribution, using
        # maximum likelihood estimate:
        # P(ti | t{i-1})
        
        """cpd_tags contains transition probabilities"""
        self.cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
        
        """print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
        print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
        print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))
        """

    
    def save_model_to_file(self, filename, cont, pickle_protocol):
        if os.path.isfile(filename):
            wstring = filename + " already exists."
            if not cont:
                wstring += " Model not saved"
                warnings.warn(wstring)
                return
            else:
                wstring += " Overwriting."
                warnings.warn(wstring)
        
        """
            Protocol version 0 is the original “human-readable” protocol and is backwards compatible with earlier versions of Python.
            Protocol version 1 is an old binary format which is also compatible with earlier versions of Python.
            Protocol version 2 was introduced in Python 2.3. It provides much more efficient pickling of new-style classes. Refer to PEP 307 for information about improvements brought by protocol 2.
            Protocol version 3 was added in Python 3.0. It has explicit support for bytes objects and cannot be unpickled by Python 2.x. This is the default protocol, and the recommended protocol when compatibility with other Python 3 versions is required.
            Protocol version 4 was added in Python 3.4. It adds support for very large objects, pickling more kinds of objects, and some data format optimizations. Refer to PEP 3154 for information about improvements brought by protocol 4.
        """
        with open(filename, "wb") as fp1:
            pickle.dump((self.cpd_tags, self.cpd_tagwords, self.tags_list), fp1, protocol = pickle_protocol)

            
    def load_model_from_file(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename + " not found")
            return
        
        with open(filename, "rb") as fp:
            self.cpd_tags, self.cpd_tagwords, self.tags_list = pickle.load(fp)

    def load_corpus_from_file(self, filename):
        lc = []
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename + " not found")
            return
        
        with open(filename, "r") as fp:
            lines_list = list(fp)
            
            for i in range(len(lines_list)):
                line = lines_list[i]
                if line == "\n":
                    continue
                    
                if ',' not in line:
                    raise RuntimeError("Every line should be comma separated.")
                    return
                
                split_sent = line.strip().split(",")
                if len(split_sent) != 2:
                    raise RuntimeError("Line " + str(i + 1) + ". Error: Every line should have exactly 2 commas.")
                    return
                
                lc.append((split_sent[0].strip(), split_sent[1].strip()))
        
        self.customcorpus = lc
    #sents is a string
    #This function returns a list of tags
    def tag(self, sents):
        ipsent = nltk.word_tokenize(sents)
        
        if self.cpd_tags is None or self.cpd_tagwords is None or self.tags_list is None:
            raise RuntimeError("Model is not trained!")
            return
        
        tags_list = self.tags_list
        cpd_tags = self.cpd_tags
        cpd_tagwords = self.cpd_tagwords
        
        seqscore = np.zeros(shape=(len(tags_list), len(ipsent)))
        backptr = np.zeros(shape=(len(tags_list), len(ipsent)))
        
        #1 is Backptrs, 0 is Seqscore
        
        #Initialization
        
        for i in range(len(tags_list)):
            pb = cpd_tagwords[tags_list[i]].prob(ipsent[0])
            if pb != 0:
                seqscore[i, 0] = pb
            else:
                seqscore[i, 0] = 1e-6
            
            backptr[i, 0] = 0
        
        for t in range(1, len(ipsent)):
            for i in range(len(tags_list)):
                
                tmp_list = []
                for j in range(len(tags_list)):
                    pb = cpd_tags[tags_list[j]].prob(tags_list[i])
                    if pb != 0:
                        tmp_list.append(seqscore[j, t - 1] * pb)
                    else:
                        tmp_list.append(seqscore[j, t - 1] * 1e-6)
                #tmp_list = [(seqscore[j, t - 1] *  cpd_tags[tags_list[j]].prob(tags_list[i])) 
                    #for j in range(len(tags_list))]
                pb2 = cpd_tagwords[tags_list[i]].prob(ipsent[t])
                if pb2 != 0:
                    seqscore[i, t] = max(tmp_list) * pb2
                else:
                    seqscore[i, t] = max(tmp_list) * 1e-6
                backptr[i, t] = tmp_list.index(max(tmp_list))
        
        
        tags = [0] * len(ipsent)
        
        tags[-1] = np.argmax(seqscore[:, -1])
        
        for i in range((len(ipsent) - 2), -1, -1):
            tags[i] = int(backptr[tags[i + 1]][i + 1])
        
        tags_final = [""] * len(tags)
        for i in range(len(tags)):
            tags_final[i] = tags_list[tags[i]]
            

        return tags_final