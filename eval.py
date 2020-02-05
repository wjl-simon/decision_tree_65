##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: 
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """
    
    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """
        
        if not class_labels:
            class_labels = list(set(prediction.tolist()))
            class_labels.sort(key=prediction.tolist().index)
            #print(class_labels)
                                    
        #transfer to label index
        prediction = [class_labels.index(x) for x in prediction]
        annotation = [class_labels.index(x) for x in annotation]
        print(prediction)
        print(annotation)
       
        size=len(class_labels)
      
        #initialize the confusion                        
        confusion = np.zeros((size,size), dtype=np.int)
        
        
        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################
        for i in range(len(prediction)):
                    confusion[annotation[i]][prediction[i]]+=1
                                
            
        return confusion
    
    
    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
 
        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################

        accuracy=0.0
        length=confusion.shape[0]
        sum_all=0
        accuracy_sum=0.0
             
        for i in range(length):
            for j in range(length):
        # accumulate the diagonol values as sum of accurate predictions
                sum_all+=confusion[i][j]
                if i==j:
                    accuracy_sum += confusion[i][i]
    
        accuracy=accuracy_sum/sum_all

        return accuracy
        
    
    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """
        
        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion), ))
     
        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        # You will also need to change this
        macro_p = 0
        sum_precision=0
        
        for c in range(len(p)):
            TP=confusion[c][c]
            sum_positive=0.0
            for r in range(len(p)):
                sum_positive+=confusion[r][c]
            p[c]=TP/sum_positive
            sum_precision+=p[c]
                
        macro_p=sum_precision/len(p)
                                                                                        

        return (p, macro_p)
    
    
    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################
        
        # You will also need to change this        
        macro_r = 0
        sum_recall=0.0
        for row in range(len(r)):
            TP=confusion[row][row]
            sum_row=0.0
            
            for column in range(len(r)):
                sum_row+=confusion[row][column]

            r[row]=TP/sum_row
            sum_recall+=r[row]
                
        macro_r=sum_recall/len(r)
                                                                                                        
        return (r, macro_r)
    
    
    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """
        
        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion), ))
        
        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################
        p=self.precision(confusion)
        r=self.recall(confusion)
        macro_f = 0
        sum_f=0.0
        length=len(confusion)
        
        for i in range(length):
            f[i]=2*(p[0][i]*r[0][i])/(p[0][i]+r[0][i])
            sum_f+=f[i]                
        

        macro_f=sum_f/length
        
        return (f, macro_f)
   
         
  
