from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import json

def rules(clf, features, labels, node_index=0,node_parent = 'null'):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    count_labels = list(zip(clf.tree_.value[node_index, 0], labels))
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        node['partition'] = json.dumps({label:count for count,label in count_labels})
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
        node['parent'] = node_parent
        
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {:g}'.format(feature, threshold)
        node['parent'] = node_parent
        node['partition'] = json.dumps({label:count for count,label in count_labels})
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index,node['name']),
                            rules(clf, features, labels, left_index,node['name'])]
    return node


data = load_iris()

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(data.data, data.target)

rules(clf, data.feature_names, data.target_names)

r = rules(clf, data.feature_names, data.target_names)
with open('rules.json', 'w') as f:
    f.write(json.dumps(r))
