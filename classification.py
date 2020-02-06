##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
# Modified by: Wenjun Li
# Your tasks: Complete the train() and predict() methods of the 
# DecisionTreeClassifier 
##############################################################################

import numpy as np
from node import Node


class DecisionTreeClassifier(object):
    """
    A decision tree classifier
    
    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    
    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    
    """

    def __init__(self):
        self.is_trained = False
        self.model = None



    def __splitDataSet(self, X, Y, node):
    # Partitions a dataset according to the node.decide() to see if it 
    # meets the rule. if so, this example will be in pos subset', otherwise,
    # in neg subset
    # @X: feature set
    # @Y: label set
    # @node: the decision tree node
        pos_X, pos_Y = [], [] # positive subset of the training set
        neg_X, neg_Y = [], [] # negative subset of the training set
        counter = 0 # the i-th example

        for example in X:
            if node.decide(example):
                pos_X.append(example)
                pos_Y.append(Y[counter])
            else:
                neg_X.append(example)
                neg_Y.append(Y[counter])
            counter += 1

        # pos_X = np.asarray(pos_X)
        # pos_Y = np.asarray(pos_Y)
        # neg_X = np.asarray(neg_X)
        # neg_Y = np.asarray(neg_Y)

        return pos_X, pos_Y, neg_X, neg_Y
    


    # the key for the list.sort() method
    def __sort_key(self,e):
        return e[0]
    

    def __sortAccorToFeatureVal(self,feature_vec, labels):
    # Return a 2D list, where the 1st column is the value of the feature 
    # (attribute), the 2nd col is the corrspodning label. The 2D list is 
    # sorted according to the feature value.
    # This function basically is uesed for narrow down the and find a good
    # spilt point
    #
    # @feature_vec: the vecture vector (a training set concerning one feature
    # only)
    # @labels: label set
    
        # concatenate the feature vector and the labels into a 2-D "dataset"
        dataset = list(zip(feature_vec.tolist(),labels.tolist()))
        # sort the dataset according to the val of the features
        dataset.sort(key = self.__sort_key)
        
        return dataset

    

    def __entropy(self,Y):
    # helper functions to compute entropy.
    # @Y: training label set

        # statistics for each label (class)
        label_stat = {}  # a dictionary of label -> count.
        for label in Y:
            if label in label_stat:
                label_stat[label] += 1
            else:
                label_stat[label] = 1

        # probability distribution of those classes/labels
        prob_distribution = [label_stat[label]/len(Y) \
            for label in label_stat]

        temp = np.dot(prob_distribution, np.log2(prob_distribution))
        return -1 * np.sum(temp)



    def __informationGain(self, child1, child2, parent_entpy):
    # helper function to compute infomation gain
    # @child1: a subset of the training set
    # @child2: another subset of the training set, where child1
    # + child2 = parent training set
    # parent_entpy: the entropy of the parent data set

        size1 = len(child1)
        size2 = len(child2)
        # weighted version of the entropy of children sets
        child_entpy = (size1 * self.__entropy(child1) + size2 * \
            self.__entropy(child2)) / (size1+size2)

        return parent_entpy - child_entpy



    def __findBestNode(self,X, Y):
    # Find the best rule that gives highest information gain to split the 
    # traning set
    # Returns the best node and the best split as well
    # @X, Y: training set

        maxInfoGain = 0
        best_node = None
        # best split
        best_pos_X, best_pos_Y, best_neg_X, best_neg_Y = [],[],[],[]
        parent_entpy = self.__entropy(Y) # entropy of the parent node
        K = X.shape[1]  # num of features

        for i in range(K):
            # sort the feature value
            dataset = self.__sortAccorToFeatureVal(X[:,i],Y)

            # looking for a good split point
            LEN = np.size(Y) - 1
            for j in range(LEN):
                # consider only split points that are between two examples
                # in sorted order that have different class labels
                if dataset[j][1] == dataset[j+1][1] or \
                    dataset[j][0] == dataset[j+1][0]:
                    continue

                # a candidate node with for spiltting the training set 
                node = Node(i,dataset[j][0])

                # splitting
                pos_X, pos_Y, neg_X, neg_Y = \
                    self.__splitDataSet(X,Y,node)

                if np.size(pos_Y) == 0 or np.size(neg_Y) == 0:
                    continue

                infoGain = \
                    self.__informationGain(pos_Y,neg_Y,parent_entpy)

                # undating accoding to info gain
                if infoGain > maxInfoGain:
                    maxInfoGain = infoGain
                    best_node = node
                    best_pos_X = pos_X
                    best_pos_Y = pos_Y
                    best_neg_X = neg_X
                    best_neg_Y = neg_Y
        
        best_pos_X = np.asarray(best_pos_X)
        best_pos_Y = np.asarray(best_pos_Y)
        best_neg_X = np.asarray(best_neg_X)
        best_neg_Y = np.asarray(best_neg_Y)

        return best_pos_X, best_pos_Y, best_neg_X, best_neg_Y, \
            best_node



    def __induceDecisionTree(self,X,Y):
    # Train the decision tree model (builds the decision tree).
    # @X,Y: training set

        # generate a node
        pos_X, pos_Y, neg_X, neg_Y, node = self.__findBestNode(X,Y)

        # Base case: if cannot split further or there is only one class,
        # return a leaf node with the majority label
        label_stat = {}
        for label in Y:
            if label in label_stat:
                label_stat[label] += 1
            else:
                label_stat[label] = 1

        if node == None or len(label_stat) < 2:
            majorLabel = max(label_stat, key = label_stat.get)
            # the leaf node always predict the majority class
            return Node(prediction=majorLabel, leftover_stat=label_stat)


        # default case: spilt
        # pos_X, pos_Y, neg_X, neg_Y = self.__splitDataSet(X,Y,node)

        # Recursion
        true_branch = self.__induceDecisionTree(pos_X, pos_Y)
        false_branch = self.__induceDecisionTree(neg_X, neg_Y)

        # node is the parent of these two branches
        true_branch.addParent(node)
        false_branch.addParent(node)

        # node append the children
        node.addChild(true_branch = true_branch, \
            false_branch = false_branch)

        return node



    def train(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the 
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array
        
        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        
        # the input and output
        #N = X.shape[0]
        #K = X.shape[1]
        
        # quit if the model has already been trained
        if self.is_trained:
            return self

        # the model has the root node
        self.model = self.__induceDecisionTree(x,y)
        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
        return self
    


    def classify(self,example,node = None):
    # gives a prediction for an example using the trained model
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        
        if node == None:
            node = self.model

        # Base case
        if isinstance(node, Node) and node.isLeafNode == True:
            return node.prediction

        # default case:
        if node.decide(example):
            return self.classify(example, node.true_branch)
        else:
            return self.classify(example, node.false_branch)

    
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the 
            number of attributes)
        
        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """
        
        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        
        # set up empty N-dimensional vector to store predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
    
        N = x.shape[0]
        for i in range(N):
            predictions[i] = self.classify(x[i])
    
        # remember to change this if you rename the variable
        return predictions