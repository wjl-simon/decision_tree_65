##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np
import pickle # for saving the model

from classification import DecisionTreeClassifier
from eval import Evaluator
from print_tree import printDecisionTree
from prune import *

def parseInputs(fname):
    filename = "F:\\PG CS Study\\IntroToML\\cw\\decision_trees_65\\data\\" + fname
    # filename = os.getcwd() + '/data/' + fname
    f = open (filename, 'r')
    line = f.readline()
    numArray = []
    charArray =[]
    while line:
        tempNumList = []
        result = [x.strip() for x in line.split(',')]
        for str in result:
            if(str.isalpha() == True):
                charArray.append(str)
            if (str.isdigit() == True):
                tempNumList.append(int(str))
        line = f.readline()
        numArray.append(tempNumList)
    f.close()
    # https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array?rq=1
    numpyNumArray = np.array([np.asarray(xi) for xi in numArray])
    numpyCharArray = np.array(charArray)
    # print(numpyNumArray)
    # print(numpyCharArray)

    return numpyNumArray, numpyCharArray


if __name__ == "__main__":
    print("Loading the training dataset...");
    # x = np.array([
    #         [5,7,1],
    #         [4,6,2],
    #         [4,6,3], 
    #         [1,3,1], 
    #         [2,1,2], 
    #         [5,2,6]
    #     ])
    
    # y = np.array(["A", "A", "A", "C", "C", "C"])
    
    x, y = parseInputs("train_full.txt")

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    #classifier = classifier.train(x, y)
    classifier.train(x,y)

    # save the trained model
    print("========================")
    print("Training finished. Saving the model...")
    file = open("model_with_train_full", "wb")
    pickle.dump(classifier, file)
    file.close()
    print("========================")

    # load the saved trained model
    # print("========================")
    # print("Loding the saved model...")
    # classifier = pickle.load(open("model_with_train_sub", "rb"))
    # classifier = pickle.load(open("model_with_train_full", "rb"))
    # print("========================")

    print("========================")
    print("Priting the decision tree...")
    printDecisionTree(classifier.model)
    print("========================")

    print("Loading the test set...")
    
    # x_test = np.array([
    #         [1,6,3], 
    #         [0,5,5], 
    #         [1,5,0], 
    #         [2,4,2]
    #     ])
    
    # y_test = np.array(["A", "A", "C", "C"])
    x_test, y_test = parseInputs("test.txt")
    
    print("======================")
    print("Actual:\n {}".format(y_test))

    predictions = classifier.predict(x_test)
    print("======================")
    print("Predictions:\n {}".format(predictions))
    
    # classes = ["A", "C"];
    
    # print("Evaluating test predictions...")
    # evaluator = Evaluator()
    # confusion = evaluator.confusion_matrix(predictions, y_test)
    
    # print("Confusion matrix:")
    # print(confusion)

    # accuracy = evaluator.accuracy(confusion)
    # print()
    # print("Accuracy: {}".format(accuracy))

    # (p, macro_p) = evaluator.precision(confusion)
    # (r, macro_r) = evaluator.recall(confusion)
    # (f, macro_f) = evaluator.f1_score(confusion)

    # print()
    # print("Class: Precision, Recall, F1")
    # for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
    #     print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));
   
    # print() 
    # print("Macro-averaged Precision: {:.2f}".format(macro_p))
    # print("Macro-averaged Recall: {:.2f}".format(macro_r))
    # print("Macro-averaged F1: {:.2f}".format(macro_f))

