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
from loading import parseInputs # the parseInput is here
from prune import pruneModel, pruneSpecifiable # the pruning function
import copy

import time
start_time = time.time()

if __name__ == "__main__":
    print("Loading the training dataset...")
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
    classifier = classifier.train(x,y)

    # # save the trained model
    # print("========================")
    # print("Training finished. Saving the model...")
    # file = open("model_with_train_noisy", "wb")
    # pickle.dump(classifier, file)
    # file.close()
    # print("========================")

    # # load the saved trained model
    # print("========================")
    # print("Loding the saved model...")
    # classifier = pickle.load(open("model_with_train_noisy", "rb"))
    # print("Loding finished.")
    # print("========================")

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

    #classes = ["A", "C"];
    classes = np.unique(y_test)
    print("======================")
    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)

    print("Confusion matrix:")
    print("(Note:row classes-actual classes; column classes-predicted classes")
    print(confusion)

    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1))

    print()
    print("Macro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))

    print("--- %s seconds ---" % (time.time() - start_time))

    print("======================")
    print("Loading the validation set...")
    x_vali, y_vali = parseInputs("validation.txt")
    print("Loading the validation set finished.")

    print("======================")
    print('Pruning the decision tree...')
    classifier_copy = copy.deepcopy(classifier)
    pruneModel(classifier_copy,x_vali,y_vali)
    print('Pruning finished. Printing the decision tree...')
    printDecisionTree(classifier_copy.model)

    print("======================")
    print('Pruning the decision tree using method2...')
    classifier2 = copy.deepcopy(classifier)
    pruneSpecifiable(classifier2,x_vali,y_vali)
    print('Pruning finished. Printing the decision tree...')
    printDecisionTree(classifier2.model)

   