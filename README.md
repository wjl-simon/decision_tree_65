## CO395 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.

### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``loading.py``
	
	* Contains a funtion ``parseInput()`` for loading a training set (x,y).


- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``node.py``

	* Contains the definition code for the class ``Node`` (the nodes in the decision
	 tree).
	 
	* Used by the ``DecisionTreeClassifier`` class.


- ``print_tree.py``
	* Conatins the code for a function to display the decision tree in the command line.


- ``prune.py``
	* Conatins the code for pruning a pretrained decision tree mode.
	
	* Includes ``pruneSpecifiable()`` and ``pruneModel()``. The other functions defined
	in this file are interanlly used by the two functions.


- ``kFold.py``
	* Contains the code for performing K-fold corss-validation for trainning a model.


- ``eval.py``

	* Contains the skeleton code for the ``Evaluator`` class. Your task is to 
implement the ``confusion_matrix()``, ``accuracy()``, ``precision()``, 
``recall()``, and ``f1_score()`` methods.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods defined in ``classification.py`` and ``eval.py``.



### Instructions

The following briefly describes how to import some of the functions

#### Loading
Simply gives the **relative path** of the training set to ``parseInput()`` and it returns
the **feature matrix** ``x`` and the **label vector** ``y``.


#### Pruning
The pruning functionality has two versions of the pruning implementations.

- ``pruneSpecifiable()`` 
This function enables users to specify a desired **accuracy improvement in percentage** 
(as long as it is indeed theoreatically acheivable) as well as the **maximum prunning depth**.

The default value for **desired accuracy impovement** (in precentage ) is 0.1 (meaning 10%), 
and default **maximum prunning depth** is 5

-  ``pruneModel()``
This bacically does the same thing as ``pruneSpecifiable()`` does, but neither the accuracy 
improvement nor the maximum prunning depth is specifiable.

- Other functions
They are internally used so please don't touch them.


#### K-Fold Cross-Validation


