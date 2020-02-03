class Node:
    ''' Nodes in the decision tree
    '''

    def __init__(self, feature = None, threshold = None,\
                true_branch = None, false_branch = None, \
                parent = None, prediction = None):
    
    # @isLeafNode: True if this node is a leaf node
    # @feature: the i-th feature, the column number of the training set
    # @threshold: for the decision
    # @true_branch, false_branch: the two child nodes.
    # @parent: a reference to the parent node. The root node's parent
    # node is None
    # @prediction: for the leaf node, the class label of the majority
    # leftover examples

        if prediction != None: # a leaf node
            self.isLeafNode = True
            self.feature = None
            self.threshold = None
            self.true_branch = None
            self.false_branch = None
            self.parent = None
            self.prediction = prediction  # always predict one class
        else: # a decision node
            self.isLeafNode = False
            self.feature = feature
            self.threshold = threshold
            self.true_branch = true_branch
            self.false_branch = false_branch
            self.parent = parent
            self.prediction = None
    

    
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
        # if is_numeric(val):
        #     return val >= self.threshold
        # else:
        #     return val == self.threshold


    # adding two children nodes
    def addChild(self,true_branch = None,false_branch = None):
        self.true_branch = true_branch
        self.false_branch = false_branch
    


    # adding a reference to this node's parent
    def addParent(self, parent):
        self.parent = parent



    # to print the rule in the decision tree
    def __repr__(self):
        if not self.isLeafNode: # a decision node
            # condition = "=="
            # if is_numeric(self.threshold):
            #     condition = ">"
            return "Is %s %s %s?" % (
                "feature"+str(self.feature), ">", str(self.threshold))
        else:   # a leaf node
            return "Leaf %s." % (str(self.prediction))
