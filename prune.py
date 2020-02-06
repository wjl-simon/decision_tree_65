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
        Prune a node into a leaf

        @node: a Node object

        Return: state of the orginal node
        @saved_feature: the feature attribute of the original node
        @saved_threshold: the threshold attribute
        @true_child: a reference to its original true_branch child
        @false_child: a reference to its original false_branch child
    '''
    assert not node.isLeafNode and node.true_branch.isLeafNode and \
        node.false_branch.isLeafNode,\
            "This node is not prunable!"
    
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
    
    # turn the node into a leaf, but don't delete the children
    true_child = node.true_branch
    false_child = node.false_branch
    saved_threshold = node.threshold
    saved_feature = node.feature
    node.true_branch = None
    node.false_branch = None
    node.isLeafNode = True
    node.threshold = None
    node.feature = None
    node.prediction = finalPrediction
    node.leftover_stat = stat

    # the info about the orginal node and the two children
    return saved_feature, saved_threshold, true_child, false_child



def undoPruneLeaf(node,saved_feature,saved_threshold, \
    true_child,false_child):
    '''
        Undo the pruneLeaf(): trun a leaf node into a decision node

        @node: a leaf node object
        @saved_feature: the feature attribute of the decision node to
        be transformed into
        @saved_threshold: the threshold attribute
        @true_child: a reference to its true_branch child
        @false_child: a reference to its false_branch child
    '''
    assert node.isLeafNode and true_child.isLeafNode and \
         false_child.isLeafNode, "Cannot undo the pruning!"

    node.true_branch = true_child
    node.false_branch = false_child
    node.isLeafNode = False
    node.threshold = saved_threshold
    node.feature = saved_feature
    node.prediction = None
    node.leftover_stat = None



def pruneModel(classifier,x_vali, y_vali):
    '''
        Iteratively prune the decision tree model, unspecifiable
        This is a inplace pruning

        @classifier: a DecisionTreeClassifier object
        @x_vali, y_vali: the validation set
    '''
    childrenStack = [classifier.model] # nodes to be processed
    evaluator = Evaluator() # computing accuracy

    # the accuracy of the unpruned model
    confusion = evaluator.confusion_matrix( \
        classifier.predict(x_vali), y_vali)
    acc_cur = evaluator.accuracy(confusion)

    while childrenStack:
        # start to process the a node's children
        node = childrenStack.pop()

        # push the noneleaves into stack
        if not node.false_branch.isLeafNode:
            childrenStack.append(node.false_branch)
        if not node.true_branch.isLeafNode:
            childrenStack.append(node.true_branch)
        
        # check if this node is  prunable
        if node.true_branch.isLeafNode and node.false_branch.isLeafNode:
            # pruning
            saved_feature, saved_threshold, true_child, \
                false_child = pruneLeaf(node)
            # compute the accuracy of the current model
            confusion = evaluator.confusion_matrix ( \
                classifier.predict(x_vali), y_vali)
            acc_next = evaluator.accuracy(confusion)
            # undo if gives worse accuracy
            if acc_next < acc_cur:
                undoPruneLeaf(node,saved_feature, \
                    saved_threshold, true_child, false_child)
            else:
                del true_child
                del false_child
                acc_cur = acc_next
    
    print('The final pruned model has an accuracy \
        of: {a}'.format(a=acc_cur))
    
    return acc_cur



#################################
# Althernative way to do pruning
#################################
def pruneSpecifiable(classifier,x_vali, y_vali,acc_improvement = 0.05, maxdep = 6):
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
    # confusin matrix
    confusion = evaluator.confusion_matrix( \
        classifier.predict(x_vali), y_vali)
    # the unpruned model's accuracy
    original_accuracy = evaluator.accuracy(confusion)

    # iteration: stops if there is a root node left, or converges, or
    # reaches max ite steps
    acc_next = original_accuracy
    acc_cur = 0
    counter = 0
    while counter < maxdep and math.fabs(acc_next - acc_cur) >= 0.01 and \
        1 - (acc_next/original_accuracy) < acc_improvement:
        # update
        acc_cur = acc_next
        counter += 1
        # commit one prunning
        acc_next = pruneModel(classifier,x_vali,y_vali)
    
    print('After {0:d} steps, the final pruned model has an accuracy \
        of: {a}'.format(counter,a=acc_next))
