# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader


"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    #print(len(X),'X')
    pos_vocab = Counter()
    neg_vocab = Counter()

    for index in range(len(y)):
        if y[index] == 1:
            pos_vocab.update(X[index])
        else:
            neg_vocab.update(X[index])

    ##TODO:

    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    ##TODO:
    new_X = []
    for email_index in range(len(y)):
        new_X.append([])
        for word_index in range(len(X[email_index])-1):
            new_X[email_index].append(X[email_index][word_index] + " " + X[email_index][word_index + 1])

    pos_vocab = Counter()
    neg_vocab = Counter()

    for index in range(len(y)):
        if y[index] == 1:
            pos_vocab.update(new_X[index])
        else:
            neg_vocab.update(new_X[index])

    uni_pos_vocab, uni_neg_vocab = create_word_maps_uni(X, y, None)

    return {**dict(pos_vocab), **uni_pos_vocab}, {**dict(neg_vocab), **uni_neg_vocab}



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels, None)
    total_1 = sum(pos_vocab.values())
    total_0 = sum(neg_vocab.values())
    pos_prob_dict = {}
    for key in pos_vocab:
        pos_prob_dict[key] = (pos_vocab[key] + laplace) / (total_1 + laplace * (1 + len(pos_vocab)))

    neg_prob_dict = {}
    for key in neg_vocab:
        neg_prob_dict[key] = (neg_vocab[key] + laplace) / (total_0 + laplace * (1 + len(neg_vocab)))

    # output dev
    retval = []
    for email in dev_set:
        prob_1 = np.log(pos_prior)
        prob_0 = np.log(1 - pos_prior)
        for word in email:
            if word not in neg_vocab:
                prob_0 += np.log((0 + laplace) / (total_0 + laplace * (1 + len(neg_vocab))))
            else:
                prob_0 += np.log(neg_prob_dict[word])
            if word not in pos_vocab:
                prob_1 += np.log((0 + laplace)/ (total_1 + laplace * (1 + len(pos_vocab))))
            else:
                prob_1 += np.log(pos_prob_dict[word])

        print(prob_1, prob_0)
        if prob_1 >= prob_0:
            retval.append(1)
        else:
            retval.append(0)
        #print(retval)


    print_paramter_vals(laplace,pos_prior)

    #raise RuntimeError("Replace this line with your code!")
    return retval


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    bi_pos_vocab, bi_neg_vocab = create_word_maps_bi(train_set, train_labels, None)
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels, None)

    #uni
    total_1 = sum(pos_vocab.values())
    total_0 = sum(neg_vocab.values())
    pos_prob_dict = {}
    for key in pos_vocab:
        pos_prob_dict[key] = (pos_vocab[key] + unigram_laplace) / (total_1 + unigram_laplace * (1 + len(pos_vocab)))

    neg_prob_dict = {}
    for key in neg_vocab:
        neg_prob_dict[key] = (neg_vocab[key] + unigram_laplace) / (total_0 + unigram_laplace * (1 + len(neg_vocab)))

    #bi
    bi_total_1 = sum(bi_pos_vocab.values())
    bi_total_0 = sum(bi_neg_vocab.values())
    bi_pos_prob_dict = {}
    for key in bi_pos_vocab:
        bi_pos_prob_dict[key] = (bi_pos_vocab[key] + bigram_laplace) / (bi_total_1 + bigram_laplace * (1 + len(bi_pos_vocab)))
    bi_neg_prob_dict = {}
    for key in bi_neg_vocab:
        bi_neg_prob_dict[key] = (bi_neg_vocab[key] + bigram_laplace) / (bi_total_0+ bigram_laplace * (1 + len(bi_neg_vocab)))

    new_dev_set = []
    for email_index in range(len(dev_set)):
        new_dev_set.append([])
        for word_index in range(len(dev_set[email_index]) - 1):
            new_dev_set[email_index].append(dev_set[email_index][word_index] + " " + dev_set[email_index][word_index + 1])

    retval = []
    for i in range(len(dev_set)):
        uni_prob_1 = np.log(pos_prior)
        uni_prob_0 = np.log(1 - pos_prior)
        bi_prob_1 = np.log(pos_prior)
        bi_prob_0 = np.log(1 - pos_prior)
        #uni
        for word in dev_set[i]:
            if word not in neg_vocab:
                uni_prob_0 += np.log((0 + unigram_laplace) / (total_0 + unigram_laplace * (1 + len(neg_vocab))))
            else:
                uni_prob_0 += np.log(neg_prob_dict[word])
            if word not in pos_vocab:
                uni_prob_1 += np.log((0 + unigram_laplace)/ (total_1 + unigram_laplace * (1 + len(pos_vocab))))
            else:
                uni_prob_1 += np.log(pos_prob_dict[word])
        #bi
        for word in new_dev_set[i]:
            if word not in bi_neg_vocab:
                bi_prob_0 += np.log((0 + bigram_laplace) / (bi_total_0 + bigram_laplace * (1 + len(bi_neg_vocab))))
            else:
                bi_prob_0 += np.log(bi_neg_prob_dict[word])
            if word not in bi_pos_vocab:
                bi_prob_1 += np.log((0 + bigram_laplace) / (bi_total_1 + bigram_laplace * (1 + len(bi_pos_vocab))))
            else:
                bi_prob_1 += np.log(bi_pos_prob_dict[word])

        if (1 - bigram_lambda) * uni_prob_0 + bigram_lambda * bi_prob_0 <= (1 - bigram_lambda) * uni_prob_1 + bigram_lambda * bi_prob_1:
            retval.append(1)
        else:
            retval.append(0)

    print(retval)

    max_vocab_size = None

    # raise RuntimeError("Replace this line with your code!")

    return retval

