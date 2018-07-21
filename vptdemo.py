# -*- coding: utf-8 -*-

from viterbitagger import ViterbiPOSTagger
from nltk.corpus import brown

postagger = ViterbiPOSTagger()
postagger.train(brown.tagged_sents()[:50])