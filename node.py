##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 nodes for the decision tree
# Prepared by: Wenjun Li
##############################################################################

class Node:
    ''' Nodes in the decision tree
    '''

    def __init__(self, feature = None, threshold = None,\
                true_branch = None, false_branch = None, \
                parent = None, prediction = None,\
                leftover_stat = None):
    # @isLeafNode: True if this node is a leaf node
    # @feature: the i-th feature, the column number of the training set
    # @threshold: for the decision
    # @true_branch, false_branch: the two child nodes.
    # @parent: a reference to the parent node. The root node's parent
    # node is None
    # @prediction: for the leaf node, the class label of the majority
    # leftover examples
    # @leftover_stat: a python dictionary for a leaf node, gives the 
    # statistics of the leftover dataset. e.g. {'class A': 500, 
    # 'class B': 600}, while the prediction is 'class B'.

        if prediction != None: # a leaf node
            self.isLeafNode = True
            self.feature = None
            self.threshold = None
            self.true_branch = None
            self.false_branch = None
            self.parent = None
            self.prediction = prediction  # always predict one class
            self.leftover_stat = leftover_stat
        else: # a decision node
            self.isLeafNode = False
            self.feature = feature
            self.threshold = threshold
            self.true_branch = true_branch
            self.false_branch = false_branch
            self.parent = parent
            self.prediction = None
            self.leftover_stat = None
    

    
    def decide(self, example):
    # method in the decision nodes, to decide an example if it belongs
    # to postive set or negative set
    # @example: an example (feature vector) of the training set
    
        # Compare the feature value in an example to the feature value 
        # in this rule.
        if self.isLeafNode:
            return self.prediction

        val = example[self.feature]
        return val >= self.threshold


    
    def addChild(self,true_branch = None, false_branch = None):
    # adding two children nodes
    # @true_branch, false_branch: the two children of this node

        self.true_branch = true_branch
        self.false_branch = false_branch
    

    
    def removeChild(self):
    # remove children
        assert self.isLeafNode == False and self.true_branch != None and\
                self.false_branch != None,\
                 "The children of a node must be both non-None."

        del self.true_branch
        del self.false_branch
        self.true_branch = None
        self.false_branch = None


   
    def addParent(self, parent):
    # adding a reference to this node's parent
        self.parent = parent


    
    def __repr__(self):
    # used when printing the rule of this node
        if not self.isLeafNode: # a decision node
            return "Is %s %s %s?" % (
                "feature"+str(self.feature), ">", str(self.threshold))
        else:   # a leaf node
            return "Leaf %s." % (str(self.prediction))
