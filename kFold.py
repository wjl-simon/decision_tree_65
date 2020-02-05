import classification
from eval import Evaluator
import numpy as np
import math
import loading
from print_tree import printDecisionTree
import example_main
import sys
from collections import Counter

np.set_printoptions(threshold=sys.maxsize)

# REMEMBER that annotation is the array containing the ground truth
#***************************************************
# split into discrete folds in accordance with foldCount (e.g. 10 columns, fold count 2 = 5, 5)
# e.g. 9 columns 2 fold count = 5, 4 each
def k_fold_cross_split(input, annotation, foldCount):
    # randomly shuffle BOTH arrays; elegant method credit to: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(input) == len(annotation), print("Length of input and annotation are not the same for K-fold Cross Validation.")
    assert foldCount <= len(input), print("Fold count cannot be bigger than number of entries. ")
    randNumber = np.random.choice(len(input), len(input), replace = False)
    # print(randNumber)
    assert foldCount >= 3, print("Fold Count must be bigger than or equal to 3.")
    shuffledInput = input[randNumber]
    shuffledAnnotation = annotation[randNumber]
    inputArraySize = np.size(annotation) - 1
    # print("input array size: " + str(inputArraySize))
    # Obtain number of columns
    testFoldSize = math.ceil(np.size(shuffledInput, 0)/foldCount)
    testFoldCounter = foldCount
    # print("shuffledInputSize: " + str(np.size(shuffledInput, 0)))
    testMin = 0
    testMax = 0 + testFoldSize - 1
    #list structure to store each fold
    # print(shuffledInput)
    # print(shuffledAnnotation)
    setTest = []
    setLabel = []
    # print("test fold size: " + str(testFoldSize))
    for foldIterator in range(testFoldCounter):
        # Elegant guide to array slicing courtesy of: https://stackoverflow.com/questions/35593187/remove-range-of-columns-in-numpy-array
         setTest.append(shuffledInput[testMin:testMax + 1,:])
         setLabel.append(shuffledAnnotation[testMin:testMax + 1])
         # print(testMin)
         # print(testMax)
         # print("testMin: " + str(testMin) + "   testMax: " + str(testMax))
         # print("shuffled annotation: ")
         # print(shuffledAnnotation)
         # print("shuffled input: ")
         # print(shuffledInput)
         # increment the min and max test set to be extracted. NEED some way to stop when it hits max.
         # print("foldIterator: " + str(foldIterator) + "    testFoldCounter: " + str(testFoldCounter))
         if (foldIterator + 2) >= testFoldCounter:
            # print("got here")
            testMin = testMax + 1
            newTestMin = inputArraySize - testFoldSize
            # print("input array size: " + str(inputArraySize) + "test fold size: " + str(testFoldSize))
            testMax = inputArraySize
            # print("newTestMin" + str(newTestMin) + "testMin" + str(testMin))
            if newTestMin > testMin:
                testMin = newTestMin
            if testMin > testMax:
                testMin = testMax
         else:
            testMin += testFoldSize
            testMax += testFoldSize
         # print(setTest)
    # print(setTest)
    # print(setLabel)
    return(setTest, setLabel)

def k_fold_cross_models(setData, foldCount, numberOfEntries):
    assert foldCount >= 3, print("Fold count must be bigger than or equal to 3.")
    assert foldCount <= numberOfEntries, print("Fold count cannot be bigger than number of entries. ")
    # print("set data: ")
    # print(setData[0])
    # print(setData[1])
    # testInput, testAnnotation = loading.parseInputs("simple1.txt")
    # print("test txt input")
    # print(testInput)
    # print("test txt output")
    # print(testAnnotation)
    TotalAnnotationNumpy = np.hstack(setData[1])
    TotalDataNumpy = np.vstack(setData[0])
    # print("total data: ")
    # print(TotalDataNumpy)
    # print("total annotation:")
    # print(TotalAnnotationNumpy)
    # print("number of entries: " + str(numberOfEntries))
    testMin = 0
    testMax = -1
    # foldCounter = math.ceil((numberOfEntries)/foldCount)
    classifierList = []
    predictionList = []
    for foldIterator in range(foldCount):
    # for foldIterator in range(0,1):
        print("======================")
        print("Model " + str(foldIterator) + " training. ")
        if (foldIterator + 1) >= (foldCount):
            testMax = numberOfEntries - 1
        else:
            testMax += (numberOfEntries//foldCount)
            # print("normal increment for test: " + str (numberOfEntries//foldCount))
        testSet = TotalDataNumpy[testMin:testMax+1, :]
        testAnnotation = TotalAnnotationNumpy[testMin:testMax+1,]
        # print("Test Min: " + str(testMin) + "Test Max:" + str(testMax))
        # print("test set: ")
        # print(testSet)
        # print("test annotation: ")
        # print(testAnnotation)
        #at start:
        # print("validation min: " + str(validationMin) + "validation max: " + str(validationMax))
        # print("validation set: ")
        # print(validationSet)
        # print("validation annotation: ")
        # print(validationAnnotation)
        # FIND OUT HOW TO DELETE ENTRIES IN ARRAY. DELETE TEST SET AND VALIDATION SET.
        # print(list(range(testMin, testMax+1), range(validationMin, validationMax+1)))
        # print("DELETE LIST: ")
        # print(deleteList)
        trainSet = np.delete(TotalDataNumpy, np.s_[testMin:testMax+1], axis = 0)
        # trainSet = np.delete(TotalDataNumpy, np.s_[testMin:testMax+1], axis = 0)
        trainAnnotation = np.delete(TotalAnnotationNumpy, np.s_[testMin:testMax+1], axis = 0)
        # trainSet = np.delete(TotalDataNumpy, np.s_[validationMin:validationMax+1], axis = 0)
        # trainAnnotation = np.delete(TotalAnnotationNumpy, np.s_[validationMin:validationMax+1], axis = 0)
        # print("train set: ")
        # print(trainSet)
        # print("train annotation: ")
        # print(trainAnnotation)
    #     print("testMin: " + str(testMin) + " testMax: " + str(testMax))
        classifier = classification.DecisionTreeClassifier()
        classifier = classifier.train(trainSet, trainAnnotation)
        # predictions = classifier.predict(testSet)
        # predictionComparison = classifier.predict(testInput)
        # printDecisionTree(classifier.model)
        classifierList.append(classifier)
        # print("Predictions:\n {}".format(predictions))
        print("======================")

    #     # print("folditerator: " + str(foldIterator) + "fold counter: " + str(foldCount))
    #     # print("test min value: " + str(testMin) + "  test max value: " + str(testMax))
        testMin = testMax + 1
    # intermediateStep = 0
    # # https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list
    # # iterate through to find the largest value
    # mostAccIndex = accuracyList.index(max(accuracyList))
    # print("The index of the tree with the highest accuracy is: " + str(mostAccIndex))
    return classifierList

def classifierMetrics(classifierList, foldCount, testInput, testAnnotation):
    accuracyList = calculateAllAccuracy(classifierList, testInput, testAnnotation)
    totalAccuracy = calculateTotalAccuracy(accuracyList, foldCount)
    standardDeviation = calculateSD(accuracyList, totalAccuracy, foldCount)
    mostAccIndex = accuracyList.index(max(accuracyList))
    return accuracyList


def calculateAllAccuracy(classifierList, testInput, testAnnotation):
    #need to find the pythonic way to do this.
    # https://stackoverflow.com/questions/44196243/iterate-over-list-of-class-objects-pythonic-way
    accuracyList = []
    for DecisionTreeClassifier in classifierList:
        classifier = DecisionTreeClassifier
        predictions = classifier.predict(testInput)
        evaluate = Evaluator()
        # print("these are my predictions.")
        # print(predictions)
        # print("these are my annotations.")
        # print(testAnnotation)
        confusion = evaluate.confusion_matrix(predictions, testAnnotation)
        accuracy = evaluate.accuracy(confusion)
        # print("Accuracy: {}".format(accuracy))
        accuracyList.append(accuracy)
    return accuracyList

def calculateTotalAccuracy(accuracyList, foldCount):
    totalAccuracy = 0.0
    listIter = 0
    print("The list of accuracies of the different models are as below: ")
    print(accuracyList)
    for accuracy in accuracyList:
        # print(accuracy)
        totalAccuracy += accuracy
        # np.add(totalAccuracy, accuracyList[listIter])
    totalAccuracy /= foldCount
    print("The average accuracy is " + str(totalAccuracy))
    return totalAccuracy

def calculateSD(accuracyList, totalAccuracy, foldCount):
    sumDiffSquare = 0.0
    for accuracy in accuracyList:
        average = accuracy
        sumDiffSquare += (average - totalAccuracy) ** 2
    variance = sumDiffSquare/foldCount
    stdDeviation = math.sqrt(variance)
    print("The standard deviation is: " + str(stdDeviation))
    return stdDeviation

def classifierPredictionsCombined(classifierList, testInput, testAnnotation):
    predictionList = []
    for DecisionTreeClassifier in classifierList:
        classifier = DecisionTreeClassifier
        predictions = classifier.predict(testInput)
        predictionList.append(predictions)
    numpyPredictions = np.vstack(predictionList)
    # CHECK that size and column are correct.
    # print(numpyPredictions)
    # print(numpyPredictions.shape)
    # print("test annotation: " + str(np.size(testAnnotation)))
    # print(testAnnotation)
    return modalAnswer(numpyPredictions)

# GET THE MODAL INPUT ASAP.
def modalAnswer(inputs):
    # print(inputs)
    dimension = inputs.shape
    rowNumber = dimension[1]
    modalAnswerList = []
    for iterator in range(0,rowNumber):
        # print(inputs[:, iterator])
        modalAnswer = modalCharacter(inputs[:, iterator])
        modalAnswerList.extend(modalAnswer)
    modalAnswer = np.array(modalAnswerList)
    # print(modalAnswer.shape)
    return np.array(modalAnswer)

def modalCharacter(input):
    # https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
    output = (Counter(input).most_common(1))[0]
    return output[0]


def mostAccurateClassifier(accuracyList):
    maxAccuracy = 0.0
    for accuracy in accuracyList:
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
    return accuracyList.index(maxAccuracy)

def outputCompared(input1, input2):
    if input1 == input2:
        return " the same as"
    if input1 > input2:
        return " better than "
    if input1 < input2:
        return " worse than "

trainInput, trainAnnotation = loading.parseInputs("simple3.txt")
foldNumber = 10
setData = k_fold_cross_split(trainInput,trainAnnotation,foldNumber)
# print(setData)
# print("size of array is: " + str(np.size(y)))
classifierList = k_fold_cross_models(setData, foldNumber, np.size(trainAnnotation))
testInput, testAnnotation = loading.parseInputs("simple4.txt")
print("============QUESTION 1============")
accuracyList = classifierMetrics(classifierList, foldNumber, testInput, testAnnotation)
print("============QUESTION 2============")
indexOfMostAccClassifier = mostAccurateClassifier(accuracyList)
print("The index of most accurate classifier is: " + str(indexOfMostAccClassifier))
mostAccClassifier = classifierList[indexOfMostAccClassifier]
mostAccEvaluate = Evaluator()
mostAccPredictions = mostAccClassifier.predict(testInput)
mostAccConfusion = mostAccEvaluate.confusion_matrix(mostAccPredictions, testAnnotation)
mostAccAccuracy = mostAccEvaluate.accuracy(mostAccConfusion)
print("Accuracy of most accurate model is: " + str(mostAccAccuracy))
fullSetClassifier = classification.DecisionTreeClassifier()
fullSetClassifier = fullSetClassifier.train(trainInput, trainAnnotation)
fullSetPrediction = fullSetClassifier.predict(testInput)
fullSetEvaluate = Evaluator()
fullSetConfusion = fullSetEvaluate.confusion_matrix(fullSetPrediction, testAnnotation)
fullSetAccuracy = fullSetEvaluate.accuracy(fullSetConfusion)
print("Accuracy of full set training is: " + str(fullSetAccuracy))
print( "The accuracy of the most accurate model is"+ outputCompared(mostAccAccuracy, fullSetAccuracy) + "training on the full dataset.")
print("============QUESTION 3============")
modalAnswer = classifierPredictionsCombined(classifierList, testInput, testAnnotation)
modalEvaluate = Evaluator()
modalConfusion = modalEvaluate.confusion_matrix(modalAnswer, testAnnotation)
modalAccuracy = modalEvaluate.accuracy(modalConfusion)
print("The accuracy of the combined predictions is: {}".format(modalAccuracy))
print( "The accuracy of modal prediction is"+ outputCompared(modalAccuracy, fullSetAccuracy) + "training on the full dataset.")

# annotation = np.array([0,1,2,3,4,5,6,7,8,9])
# input = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]])
# setData = k_fold_cross_split(input, annotation, 10)
# #k fold cross plit
# k_fold_cross_calculation(setData, 10, np.size(annotation))
