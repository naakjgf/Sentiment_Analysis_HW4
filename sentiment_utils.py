# FIRST: RENAME THIS FILE TO sentiment_utils.py 

# YOUR NAMES HERE: Kaan Tural, Arinjay Singh


"""

CS 4120
Homework 4
Spring 2024

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
# for plotting
import matplotlib.pyplot as plt
import numpy as np
# so that we can indicate a function in a type hint
from typing import Callable
nltk.download('punkt')

def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    return X, y


"""
NOTE: for all of the following functions, we have provided the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    #gonna use sklearn for this
    precision = precision_score(dev_y, preds, average='binary')
    recall = recall_score(dev_y, preds, average='binary')
    f1 = f1_score(dev_y, preds, average='binary')
    accuracy = accuracy_score(dev_y, preds)
    
    if verbose:
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("f1: " + str(f1))
        print("Accuracy:" + str(accuracy))
    
    return precision, recall, f1, accuracy

def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    dev_y = [label for feats, label in dev_feats]
    
    precision_list, recall_list, f1_list, accuracy_list = [], [], [], []
    
    # training sizes to use like 10% to 100% of training data
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    for size in train_sizes:
        num_examples = int(size * len(train_feats))
        
        #making a subset of the training data
        train_subset = train_feats[:num_examples]
        
        # Use the provided metrics_fun to evaluate the model on this subset, GPT generated this line
        precision, recall, f1, accuracy = metrics_fun(train_subset, dev_feats)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        
        if verbose:
            print(f"Training with {num_examples} examples: Precision={precision}, Recall={recall}, F1={f1}, Accuracy={accuracy}")
    
    # Plotting the metrics, GPT generated this:
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * len(train_feats), precision_list, label='Precision')
    plt.plot(train_sizes * len(train_feats), recall_list, label='Recall')
    plt.plot(train_sizes * len(train_feats), f1_list, label='F1 Score')
    plt.plot(train_sizes * len(train_feats), accuracy_list, label='Accuracy')
    plt.title(f'Model Performance as a Function of Training Data Size - {kind}')
    plt.xlabel('Size of Training Data')
    plt.ylabel('Score')
    plt.legend()
    
    if savepath:
        plt.savefig(savepath)
    plt.show()



def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    vocab = set(word for document in all_train_data_X for word in document)
    return list(vocab)


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    vocab_set = set(vocab)  # changing to a set because the comment above implies that this will take a while, so O(1)
                            # lookup will probable be useful in this case. May run into issues w/ sets not liking duplicates heads up.
    vectorized_data = []

    for document in data_to_be_featurized_X:
        document_counter = Counter(document)
        if binary:
            document_vector = [1 if word in document_counter else 0 for word in vocab]
        else:
            document_vector = [document_counter[word] if word in document_counter else 0 for word in vocab]
        vectorized_data.append(document_vector)
        
        if verbose:
            print(document_vector)
    
    return vectorized_data