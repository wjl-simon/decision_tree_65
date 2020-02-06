from classification import Node


def printDecisionTree(node, spacing="|", depth = 0, maxDep = 4):
    # base case 1: reaches the max recursion depth
    if depth > maxDep:
        print (spacing + '==* ...')
        return

    # Base case 2: reaches a leaf
    #if isinstance(node, Node) and node.isLeafNode == True:
    if node.isLeafNode == True:
        #print (spacing + "Predict", node.prediction)
        print (spacing + "==* L" + str(depth) + str(".") + str(node))
        return

    # Print the rule at this node
    print (spacing + "==* L" + str(depth) + str(". ") + str(node))

    # Call this function recursively on the true branch
    print (spacing + '---> L' + str(depth) + str(". ") + 'True:')
    printDecisionTree(node.true_branch, spacing + "-|", depth + 1)

    # Call this function recursively on the false branch
    print (spacing + "---> L" + str(depth) + str(". ") + 'False:')
    printDecisionTree(node.false_branch, spacing + "-|", depth + 1)