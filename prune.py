##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 pruning decision tree
# Prepared by: Wenjun Li
##############################################################################

from node import Node
from classification import DecisionTreeClassifier
from eval import Evaluator
import copy
import math

def pruneLeaf(node):
    '''
        For each intermediate node where all its children are leaf nodes,
        convert this node to a single leaf node (and set the class label 
        by majority vote)

        @node: the root node of the decision tree
    '''
    # base case : reaches a intermediate node where all its children
    #  are leaf nodes
    if not node.isLeafNode and node.true_branch.isLeafNode and \
        node.false_branch.isLeafNode:
        # get the statistics
        leftover_stat_false = node.false_branch.leftover_stat
        stat = node.true_branch.leftover_stat.copy() # make a copy
        # merge leftover_stat_false into the stat
        for label in leftover_stat_false:
            if label in stat:
                # add the value to the same label
                stat[label] += leftover_stat_false[label]
            else:
                stat[label] = leftover_stat_false[label]

        # voting
        finalPrediction = None
        maxSize = 0
        for label in stat:
            if stat[label] > maxSize:
                maxSize = stat[label]
                finalPrediction = label
        
        # Pruning
        node.removeChild()
        node.isLeafNode = True
        node.thershold = None
        node.feature = None
        node.prediction = finalPrediction
        node.leftover_stat = stat

        return
    # recursion: search deeper
    elif not node.isLeafNode and node.true_branch.isLeafNode and\
        not node.false_branch.isLeafNode:
        return pruneLeaf(node.false_branch)
    elif not node.isLeafNode and not node.true_branch.isLeafNode and\
        node.false_branch.isLeafNode:
        return pruneLeaf(node.true_branch)
    elif not node.isLeafNode and not node.true_branch.isLeafNode and\
        not node.false_branch.isLeafNode:
        pruneLeaf(node.true_branch)
        pruneLeaf(node.false_branch)
        return



def pruneModel(classifier,x_vali, y_vali, acc_loss_percent = 0.1,\
             maxStep = 5):
    '''
        Gives a pruned version of the decision tree
        
        @classifer: is a DecisionTreeClassifier instance
        @x_valid, y_valid: the validation set
        @acc_loss_percent: a convergence condition, specifies the 
        percentage of loss of the accuracy with respect to the unpruned
        version. (e.g. acc_loss_percent = 0.1, the accuracy of the 
        unpruned model gives 80% of accuracy, when a pruned version gives
        an accuray of less 72% (10% loss), the function returns.
        @maxStep: the maximun times of doing prunning. A maxStep > the 
        depth of the decision tree will not screw up the model.
    '''
    assert classifier.is_trained, \
        "Pruning failed. the classifier must be pretrained."

    evaluator = Evaluator()
    # prediction of the unpruned classifer
    original_predition = classifier.predict(x_vali)
    # confusin matrix
    confusion = evaluator.confusion_matrix(original_predition, y_vali)
    # the unpruned model's accuracy
    original_accuracy = evaluator.accuracy(confusion)
    # make a deep copy of the model
    classifier_copy = copy.deepcopy(classifier)

    # iteration: stops if there is a root node left, or converges, or
    # reaches max ite steps
    acc_cur = 0
    acc_next = original_accuracy
    counter = 0

    while counter < maxStep and \
        1 - math.fabs(acc_next/original_accuracy) < acc_loss_percent:
        # update
        acc_cur = acc_next
        counter += 1
        # prune the leaves
        pruneLeaf(classifier_copy.model)
        # compute the accuracy
        pred = classifier_copy.predict(x_vali)
        conf = evaluator.confusion_matrix(pred,y_vali)
        acc_next = evaluator.accuracy(conf)
        
    
    print('After {0:d} steps, the final pruned model has an accuracy \
        of: {a}'.format(counter,a=acc_cur))
    return classifier_copy

