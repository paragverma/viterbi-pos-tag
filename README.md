## Part of Speech Tagger
#### This is an easy-to-use POS Tagger written in Python3 which utilizes the Viterbi Algorithm (Hidden Markov Model) to mark/tag individual words given a sentence
Eg. Input Sentence -> *Today is a wonderful day!*<br>
            Output -> Today:**NN**[Noun],  is:**VBZ**[Verb],  a:**DT**[Determiner],  day:**NN**[Noun]

The output POS Tags depend on the tags used in the training sentences. Usually, the Penn Treebank Tagset is used. See the complete list [here](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

#### How to use:
    $python viterby.py -s "Today is a wonderful day"
    ['NN', 'VBZ', 'DT', 'NN']
There is also a trimmed version which can be imported by other projects #### Coming Soon

#### Some important notes
- The default training corpus is the *Brown* corpus. You can change it any other corpus in the *nltk* library(see list below) or supply your own corpus(see format below)
- Pre-trained models are available in the */models* folder of this repository(see usage below)
- 
## 
