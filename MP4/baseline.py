# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    retval = []
    word_dict = {}
    POS_dict = {}
    for sentence in train:
        for word, POS in sentence:
            # word dict
            if word not in word_dict:
                word_dict[word] = {POS: 1}
            else:
                if POS not in word_dict[word]:
                    word_dict[word][POS] = 1
                else:
                    word_dict[word][POS] += 1
            # POS dict
            if POS not in POS_dict:
                POS_dict[POS] = 1
            else:
                POS_dict[POS] += 1

    unseen = max(POS_dict, key=POS_dict.get)
    for test_sentence_index in range(len(test)):
        retval.append([])
        for test_word_index in range(len(test[test_sentence_index])):
            cur_word = test[test_sentence_index][test_word_index]
            if cur_word in word_dict:
                retval[test_sentence_index].append((cur_word, max(word_dict[cur_word], key=word_dict[cur_word].get)))
            else:
                retval[test_sentence_index].append((cur_word, unseen))

    return retval
