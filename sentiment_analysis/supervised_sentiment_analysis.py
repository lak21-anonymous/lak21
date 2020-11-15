import pandas as pd
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import cohen_kappa_score
from scipy.sparse import issparse

from constants import *

def get_word_counts(data):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data['Answer'])

def get_tfidf(data):
    transformer = TfidfTransformer(smooth_idf=False)
    return transformer.fit_transform(get_word_counts(data))

def random_forest_accuracy(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
def random_forest_accuracy_summary(results, key = 'Manual Tag'):
    labeled_results = results[~(results[key] == 'no tag')]
    labels = np.array(labeled_results[key])
    print('Accuracy for Regular Word Counts')
    random_forest_accuracy(get_word_counts(labeled_results), labels)
    print('Accuracy for TFIDF Normalized Word Counts')
    random_forest_accuracy(get_tfidf(labeled_results), labels)
    
    
def plot_model_confusion_matrix(features, labels, model = 'rfc',):
    # split labeled data 70-30 for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    if model == 'rfc':
        clf = RandomForestClassifier()
    elif model == 'gnb':
        clf = GaussianNB()
        # gnb doesn't like sparse arrays, the other models are ok with them
        if issparse(features):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
    elif model == 'svc':
        clf = svm.SVC()
    else:
        clf = None
        assert False
    
    # fit model and predict labels
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    
    # create confusion matrix
    class_names = ['positive', 'neutral', 'negative']
    conf_mat = confusion_matrix(y_test, y_pred, labels= class_names)
    
    #graphing stuff
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    return conf_mat

def confusion_matrix_summary(data, model = 'rfc'):
    labeled_results = data[~(data['Categorical Tag'] == 'no tag')]
    
    print('Results for Question 1')
    q1_results = labeled_results[labeled_results['Question'] == Q1]
    features = get_word_counts(q1_results)
    labels = np.array(q1_results['Categorical Tag'])
    plot_model_confusion_matrix(features, labels, model = model)
    
    print('Results for Question 2')
    q2_results = labeled_results[labeled_results['Question'] == Q2]
    features = get_word_counts(q2_results)
    labels = np.array(q2_results['Categorical Tag'])
    plot_model_confusion_matrix(features, labels, model = model)
    
    print('Results for Combined Questions')
    features = get_word_counts(labeled_results)
    labels = np.array(labeled_results['Categorical Tag'])
    plot_model_confusion_matrix(features, labels, model = model)
    
def normalized(a, axis=-1, order=2):
    l1 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l1[l1==0] = 1
    return a / np.expand_dims(l1, axis)

def get_conf_matrix(X_train, X_test, y_train, y_test, clf, normalize = False):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred, labels = CLASS_NAMES)
    if normalize:
        return normalized(conf_mat, axis = 1, order = 1)
    return conf_mat

def get_conf_matrices(features, labels, clfs = ['svc', 'gnb', 'rfc'], trials = 1, splits = 10, test_size = 0.3):
    iterations = trials * splits
    clf_dict = {
        'svc': svm.SVC(),
        'gnb': GaussianNB(),
        'rfc': RandomForestClassifier()
    }
    confusion_matrices = {}
    for clf in clfs:
        confusion_matrices[clf] = []
    
    count = 0
    for k in range(splits):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size)
        if not isinstance(X_train, np.ndarray):
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
        for j in range(trials):
            for clf in clfs:
                if clf == 'gnb' and not isinstance(X_train, np.ndarray):
                    conf_matrix = get_conf_matrix(X_train_dense, X_test_dense, 
                                                  y_train, y_test, clf_dict[clf], normalize = False)
                else:
                    conf_matrix = get_conf_matrix(X_train, X_test, 
                                                  y_train, y_test, clf_dict[clf], normalize = False)
                confusion_matrices[clf].append(conf_matrix)
            count += 1
            if ((count+1)% (iterations//10) == 0) and iterations > 20:
                print(count+1, "trials done out of", iterations)
    return confusion_matrices


def get_average_heatmaps(confusion_matrices, normalize = True, trials = 1, splits = 10, figsize = (8,8)):
    average_confusion_matrices = copy.deepcopy(confusion_matrices)
    clf_names = {
        'svc': 'Support Vector Classifier',
        'gnb': 'Gaussian Naive Bayes',
        'rfc': 'Random Forest Classifier'
    }
    l = len(confusion_matrices)
    fig, axs = plt.subplots(l, squeeze=False, figsize = (figsize[0],figsize[1] * l), gridspec_kw={'hspace': 0.2, 'wspace': 0})
    pos = 0
    for clf, matrices in average_confusion_matrices.items():
        average_confusion_matrix = sum(matrices)/len(matrices)
        if normalize:
            average_confusion_matrix = normalized(average_confusion_matrix, axis = 1, order = 1)
        print(average_confusion_matrix)
        vmin = 0 if normalize else None
        vmax = 1 if normalize else None
        ax = axs[pos,0]
        print(ax)
        pos+=1
        sns.heatmap(average_confusion_matrix, cmap=plt.cm.Blues, ax = ax, 
                    vmin = vmin, vmax = vmax, 
                    square = True, annot = True, 
                    xticklabels = CLASS_NAMES, yticklabels = CLASS_NAMES,
                    annot_kws={"fontsize":14})
        ax.set_yticklabels(rotation = 0, size = 14, labels = CLASS_NAMES)
        ax.set_xticklabels(size = 14, labels = CLASS_NAMES)
        trial_string = '' if trials == 1 else str(trials)+" trials, "
        normalized_string = 'Normalized ' if normalize else ''
        title = f"{normalized_string}Confusion Matrix using\n{clf_names[clf]}, {trial_string}{str(splits)} splits"
        ax.set_title(title, size = 18)
        ax.set_xlabel('Predicted Label', size = 16)
        ax.set_ylabel('True Label', size = 16)
    
    
def get_cross_validation_scores(features, labels, clf, cv_count = 10):
    scores = cross_val_score(clf, features, labels, cv=cv_count)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores


def get_kappa(features, labels, clf, test_size = 0.3, trials = 1):
    kappas = []
    for k in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_size)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        kappas.append(cohen_kappa_score(y_test, y_pred))
    return kappas
    
def macro_F(conf_matrix):
    dim = len(conf_matrix)
    F_scores = []
    for k in range(dim):
        correct = conf_matrix[k][k]
        false_positive = sum([conf_matrix[j][k] for j in range(dim)])-correct
        false_negative = sum([conf_matrix[k][j] for j in range(dim)])-correct
        precision = correct/(correct+false_positive)
        recall = correct/(correct+false_negative)
        F = 2*precision*recall/(precision+recall)
        F_scores.append(F)
    return np.mean(F_scores)