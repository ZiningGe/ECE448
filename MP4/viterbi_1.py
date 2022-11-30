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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import numpy as np

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    retval = []
    word_dict = {}
    POS_dict = {}
    POS_pair_dict = {}
    for sentence in train:
        for index in range(len(sentence)):
            word = sentence[index][0]
            POS = sentence[index][1]
            # word_dict
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

        for index in range(len(sentence)-1):
            cur_POS = sentence[index][1]
            next_POS = sentence[index + 1][1]
            if (cur_POS, next_POS) not in POS_pair_dict.keys():
                POS_pair_dict[(cur_POS, next_POS)] = 1
            else:
                POS_pair_dict[(cur_POS, next_POS)] += 1

    POS_distinct_dict = {}
    for POS in POS_dict:
        count = 0
        for word in word_dict.keys():
            if POS in word_dict[word]:
                count += 1
        POS_distinct_dict[POS] = count

    transition_prob = {}
    for first, second in POS_pair_dict:
        transition_prob[(first, second)] = POS_pair_dict[(first, second)] / POS_dict[first]

    emission_prob = {}
    for word in word_dict:
        for POS in POS_dict:
            if POS in word_dict[word]:
                emission_prob[(POS, word)] = (word_dict[word][POS] + 0.001) / (POS_dict[POS] + 0.001 * (1 + POS_distinct_dict[POS]))
            # else:
            #     emission_prob[(POS, word)] = (0 + 1) / (POS_dict[POS] + 1 * (1 + POS_distinct_dict[POS]))

    "-------------------------------------------------------------"
    POS_list = [POS for POS in POS_dict]
    start_index = POS_list.index("START")
    end_index = POS_list.index("END")
    number_of_POS = len(POS_dict)
    count = 0
    for sentence in test:
        # initialize
        previous = np.ones((number_of_POS, len(sentence))) * -1
        node = np.zeros((number_of_POS, len(sentence)))
        previous[:, 1] = start_index
        for i in range(len(POS_list)):
            if i == start_index:
                node[i][0] = np.log(1)
            else:
                node[i][0] = np.log(0)
        #print(previous)
        # process
        for j in range(1, len(sentence)-1):
            for k in range(len(POS_list)): # cur POS
                max = np.log(0)
                max_index = -1
                for h in range(len(POS_list)): # previous POS
                    if (POS_list[h], POS_list[k]) not in transition_prob:
                        t = 0
                    else:
                        t = transition_prob[(POS_list[h], POS_list[k])]
                    if (POS_list[k], sentence[j]) not in emission_prob:
                        e = (0 + 0.001) / (POS_dict[POS_list[k]] + 0.001 * (1 + POS_distinct_dict[POS_list[k]]))
                    else:
                        e = emission_prob[(POS_list[k], sentence[j])]

                    temp = node[h][j-1] + np.log(t) + np.log(e)
                    #print(node[h][j-1], np.log(t), np.log(e), temp)
                    if temp > max:
                        max = temp
                        max_index = h
                node[k][j] = max
                previous[k][j] = max_index


        cur = int(np.argmax(node[:, -2]))
        ret = [POS_list[end_index], POS_list[cur]]
        index = len(sentence)-2
        while previous[cur][index] != start_index:
            ret.append(POS_list[int(previous[cur][index])])
            cur = int(previous[cur][index])
            index -= 1
        ret.append(POS_list[start_index])
        ret.reverse()

        ret_sentence = []
        for index in range(len(sentence)):
            ret_sentence.append((sentence[index], ret[index]))
        retval.append(ret_sentence)
        #print(count)
        count +=1
    print(retval)

    return retval