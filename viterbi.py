import nltk
import sys
from nltk.corpus import brown
import numpy as np

# Estimating P(wi | ti) from corpus data 
# P(wi | ti) = count(wi, ti) / count(ti)
#
# We add an artificial "start" tag at the beginning of each sentence, and
# We add an artificial "end" tag at the end of each sentence.
# So we start out with the brown tagged sentences,
# add the two artificial tags,
# and then make one long list of all the tag/word pairs.

brown_tags_words = [ ]
i = 0
for sent in brown.tagged_sents()[0:100]:
    print("Training: {}".format(i))
    i += 1
    # sent is a list of word/tag pairs
    # add START/START at the beginning
    brown_tags_words.append( ("START", "START") )
    # then all the tag/word pairs for the word/tag pairs in the sentence.
    # shorten tags to 2 characters each
    brown_tags_words.extend([ (tag[:2], word) for (word, tag) in sent ])
    # then END/END
    brown_tags_words.append( ("END", "END") )

# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

"""print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))
"""

# Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
# P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
brown_tags = [tag for (tag, word) in brown_tags_words ]
tags_list = list(set(brown_tags))

# make conditional frequency distribution:
# count(t{i-1} ti)
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# make conditional probability distribution, using
# maximum likelihood estimate:
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

"""print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))
"""


ipsent = str(input("Enter sentence: "))
print("\nMissing Bigrams and Transition probabilities assumed 1e-6\n")

ipsent = nltk.word_tokenize(ipsent)

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
    
print(ipsent)
print(tags_final)