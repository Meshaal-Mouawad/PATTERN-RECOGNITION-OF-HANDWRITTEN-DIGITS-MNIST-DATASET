"""
    This module provides functions to compute a confusion
    matrix and print results.

    15 Aug 2020
    John Ball

"""
import numpy as np


#_______________________________________________________________________________________________________________________
#
# Function to create a confusion matrix.
#
#   The rows of the CM are true classes. The columns are predicted classes.
#
#   Inputs:
#   true          Length N numpy vector of true class labels.
#   pred          Length N numpy vector of predicted class labels.
#
#   Outputs:
#   cm             A num_classes X num_classes numpy confusion matrix.
#                  Row are true classes, columns are predictions.
#
#   acc            Accuracy as a fraction, e.g. in [0,1]. 1 is 100 %
#
#       Confusion matrix is CM[true, pred]
#
#       Example for a two-class problem.
#
#                Predict=0   Predict=1
#              +-----------+-----------+
#      True=0  |  cm[0,0]  |  cm[0,1]  |
#              +-----------+-----------+
#      True=1  |  cm[1,0]  |  cm[1,1   |
#              +-----------+-----------+
#
#                Predict=0   Predict=1
#              +-----------+-----------+
#      True=0  |     TN    |     FP    |
#              +-----------+-----------+
#      True=1  |     FN    |     TP    |
#              +-----------+-----------+
#_______________________________________________________________________________________________________________________
#
def confusion_matrix(true, pred):

    true = true.flatten().astype('int32')
    pred = pred.flatten().astype('int32')

    all_labels = np.concatenate((true, pred))
    unique_labels = np.matrix(np.unique(all_labels))
    num_classes = unique_labels.size

    cm = np.zeros((num_classes, num_classes))
    num_labels = true.size

    for k in np.arange(num_labels):
        rr = true[k]
        cc = pred[k]
        cm[rr, cc] += 1

    acc = np.matrix.trace(cm) / num_labels
    return cm, acc


#_______________________________________________________________________________________________________________________
#
# Print the confusion matrix
#_______________________________________________________________________________________________________________________
#
def print_confusion_matrix(cm, label, print_final_space=False, true="", percent=False):

    # Verify square matrix
    if cm.shape[0] != cm.shape[1]:
        raise ValueError('Confusion Matrix must be square.')

    # Get number of classes
    num_classes = cm.shape[0]

    if not percent:
        print("\n" + label + " Confusion Matrix: (Rows are true classes, columns predictions)\n")
        cm = np.array(cm).astype('int32')
        for r in np.arange(num_classes):
            cstr = ""
            for c in np.arange(num_classes):
                cstr = cstr + '{:-9d}'.format(cm[r, c]) + " "
            print("%s" % cstr)
        if print_final_space:
            print("\n")
    else:
        # Print as percentage
        print("\n" + label + " Confusion Matrix in percent: (Rows are true classes, columns predictions)\n")
        cm = np.array(cm).astype('float32')
        cm_rowsum = cm.sum(axis=1)
        for r in np.arange(num_classes):
            # Convert to percentage (remember each row is the true class)
            cm[r, :] = 100.0 * cm[r, :] / cm_rowsum[r]
            cstr = ""
            for c in np.arange(num_classes):
                cstr = cstr + '{:-10.6f}'.format(cm[r, c]) + "  "
            print("%s" % cstr)
        if print_final_space:
            print("\n")


#_______________________________________________________________________________________________________________________
#
# Get stats from confusion matrix
# See https://en.wikipedia.org/wiki/Confusion_matrix
#
# Input:
#    cm          Confusion matrix.
#    pos         Positive class. Only applies for a 2X2 confusion matrix. Set to 0 or 1. Defaults to 1.
#
# Output:
#    oa          Overall accuracy in [0,1].
#
#                The following stats only applies to a two-class confusion matrix. If this isn't a two-class
#                problem, then these are set to [].
#
#    tpr         True positive rate in [0,1].
#    fpr         False positive rate in [0,1].
#    tnr         True negative rate in [0,1].
#    fnr         False negative rate in [0,1].
#_______________________________________________________________________________________________________________________
#
def get_confusion_matrix_stats(cm, pos=1):

    # Get total entries and numbers from CM for TN, TP, FN, FP
    total_entries = np.sum(cm)

    # Overall accuracies
    oa = np.matrix.trace(cm) / total_entries  # Overall accuracy

    #
    #  Confusion matrix is CM[true, pred]
    #
    #            Predict=0   Predict=1
    #          +-----------+-----------+
    #  True=0  |  cm[0,0]  |  cm[0,1]  |
    #          +-----------+-----------+
    #  True=1  |  cm[1,0]  |  cm[1,1   |
    #          +-----------+-----------+
    #
    #  Example with class 1 as the target class.
    #
    #           Predict=0   Predict=1
    #          +-----------+-----------+
    #  True=0  |     TN    |     FP    |
    #          +-----------+-----------+
    #  True=1  |     FN    |     TP    |
    #          +-----------+-----------+
    #
    #  Example with class 0 as the target class.
    #
    #           Predict=0   Predict=1
    #          +-----------+-----------+
    #  True=0  |     TP    |     FN    |
    #          +-----------+-----------+
    #  True=1  |     FP    |     TN    |
    #          +-----------+-----------+
    #

    if cm.shape[0] == 2 and cm.shape[1] == 2:
        
        if pos == 1:
            # Get TP, TN, FP, FN from CM with positive class = 1
            tn = cm[0, 0]  # True=0, Pred=0 is TN
            fp = cm[0, 1]  # True=0, Pred=1 is FP
            fn = cm[1, 0]  # True=1, Pred=0 is FN
            tp = cm[1, 1]  # True=1, Pred=1 is TP
        elif pos == 0:
            # Get TP, TN, FP, FN from CM with positive class = 0
            tp = cm[0, 0]  # True=0, Pred=0 is TP
            fn = cm[0, 1]  # True=0, Pred=1 is FN
            fp = cm[1, 0]  # True=1, Pred=0 is FP
            tn = cm[1, 1]  # True=1, Pred=1 is TN
        else:
            raise ValueError('Positive class must be 0 or 1.')

        # Calculate TRP, FPR, TNR, FNR
        if (tp + fn) > 0:
            tpr = tp / (tp + fn)   # True positive rate
        else:
            tpr = 0.0
            
        if (tn + fp) > 0:
            tnr = tn / (tn + fp)   # True negative rate
        else:
            tnr = 0.0
            
        if (fn + tp) > 0:
            fnr = fn / (fn + tp)   # False negative rate
        else:
            fnr = 0.0
    
        if (fp + tn) > 0:   
            fpr = fp / (fp + tn)   # False positive rate
        else:
            fpr = 0.0
        
    else:
        tpr = []
        fpr = []
        tnr = []
        fnr = []
        
    return oa, tpr, fpr, tnr, fnr


#_______________________________________________________________________________________________________________________
#
# Print confusion matrix stats
#
# cm        Confusion matrix. If the confusion matrix is a two-class matrix, then TPR, FPR, TNR and FNR are printed.
# label     'Training', 'Testing', etc. A string to label the matrix.
# pos       Positive class. Only applies for a 2X2 confusion matrix. Set to 0 or 1. Defaults to 1.
# percent   False to print CM counts (integers), True to print CM as percent.
#_______________________________________________________________________________________________________________________
#
def print_confusion_matrix_stats(cm, label, pos=1, percent=False):

    oa, tpr, fpr, tnr, fnr = get_confusion_matrix_stats(cm, pos=pos)

    print_confusion_matrix(cm, label=label, percent=percent)
    print("\nConfusion matrix %s stats:" % label)
    print("Total instances           : %d" % np.sum(cm))
    print("Overall accuracy    (OA)  : %-7.5f" % oa)
    
    if cm.shape[0] == 2 and cm.shape[1] == 2:
        print("True positive rate  (TPR) : %-7.5f" % tpr)
        print("True negative rate  (TNR) : %-7.5f" % tnr)
        print("False positive rate (FPR) : %-7.5f" % fpr)
        print("False negative rate (FNR) : %-7.5f" % fnr)
    print("")


#_______________________________________________________________________________________________________________________
#
# This function tests some cases to verify the CM code.
#_______________________________________________________________________________________________________________________
#
def test_cm():

    # 2-class example
    label = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).astype('int32')
    pred =  np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1]).astype('int32')
    cm, _ = confusion_matrix(label, pred)
    print_confusion_matrix_stats(cm, '2X2 pos=1')
    print_confusion_matrix_stats(cm, '2X2 pos=0', pos=0)
    
    # 3-class example
    label = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]).astype('int32')
    pred =  np.array([0, 0, 1, 2, 1, 1, 0, 0, 2, 1, 2, 2, 0]).astype('int32')
    cm, _ = confusion_matrix(label, pred)
    print_confusion_matrix_stats(cm, '3X3')
