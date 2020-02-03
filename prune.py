def pruning(node):
    '''
        pruning all the leaf nodes. For each intermediate node where
        all its children are leaf nodes, convert this node to a 
        single leaf node (and set the class label by majority vote)

        @node: the root node of the decision tree
    '''
    return