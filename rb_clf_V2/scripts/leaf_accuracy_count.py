import sys
sys.path.insert(0, '../..')

from rb_clf_V2.scripts.utils import plot_config
from rb_clf_V2.scripts.train_RBclf import load_single, get_akb_labels
import time
from coniferest.isoforest import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
#from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import pandas as pd




def get_node_path(n_tree, node, iforest):
    #tree = iforest.estimators_[n_tree].tree_ # sklearn
    tree = iforest.trees[n_tree]  #coniferest
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    if children_left[node] != children_right[node]:
        print(f"Node {node} is not a leaf!")
        return None

    # 2. Поднимаемся от node к корню, собирая путь
    path = [(node, feature[node], threshold[node], 'leaf')]
    current_node = node
    while current_node != 0:  # пока не дошли до корня
        # Определяем, был ли current_node левым или правым потомком
        if current_node in children_left:
            direction = "left"
            parent_node = np.where(children_left == current_node)[0][0]
        else:
            direction = "right"
            parent_node = np.where(children_right == current_node)[0][0]
        # Записываем (родитель, признак, порог, направление)
        path.append((parent_node, feature[parent_node], threshold[parent_node], direction))
        current_node = parent_node

    # 3. Разворачиваем путь, т.к. шли от node к корню, а нужно от корня к node
    path = path[::-1]

    # Выводим путь
    # print(f'Tree index: {n_tree}')
    # print(f"len(Path) of node {node}: {len(path) - 1}")
    # print(f"Path to node {node}:")
    # for step in path[:-1]:
    #     node_id, feat, thr, direction = step
    #     #right -- False, left -- True
    #     print(f"Node {node_id}: split X[:, {feat}] <= {thr:.3f}, go {direction}")
        
    # node_id, feat, thr, direction = path[-1]
    # print(f"Node {node_id} is a leaf")
    return len(path) - 1


def get_forest_acc(iforest, data, labels):
    total_acc, total_nodes = [], []
    # for estimator in iforest.estimators_: # for sklearn version
    #     tree = estimator.tree_
    for tree in iforest.trees:
        tree_acc = []
        nodes = []
        tree_out = tree.apply(data)
        for cur_node in np.unique(tree_out):
            cur_acc = ((tree_out == cur_node).astype(int) == labels).sum() / len(labels)
            tree_acc.append(cur_acc)
            nodes.append(cur_node)
        tree_acc = np.array(tree_acc)
        total_acc.append(tree_acc)
        total_nodes.append(nodes)
    return total_acc, total_nodes

def get_forest_nodes(iforest, data):
    nodes = []
    # for estimator in iforest.estimators_: # for sklearn version
    #     tree = estimator.tree_
    for tree in iforest.trees:
        tree_out = tree.apply(data)
        nodes.append(tree_out)
    return nodes

def get_tree_id(idx, total_acc, total_nodes):
    cur_tree_id = 0
    cur_length = len(total_acc[cur_tree_id])
    while cur_length < idx + 1:
        cur_tree_id += 1
        cur_length += len(total_acc[cur_tree_id])
    idx_in_tree = idx - (cur_length - len(total_acc[cur_tree_id]))
    acc = total_acc[cur_tree_id][idx_in_tree]
    node = total_nodes[cur_tree_id][idx_in_tree]
    #print(f'Accuracy score: {acc:.3f}')
    return acc, node, cur_tree_id




oids, features = load_single('../../dr23-features/sid_snad_clf_r_100.dat', '../../dr23-features/feature_rb.dat')

akb = get_akb_labels(f'../data/akb.ztf.snad.space.json')
crossmatch = np.load(f'../data/crossmatch.npy')
akb_features = features[crossmatch] 
akb_labels = 1 - np.array([akb[oid] for oid in oids[crossmatch]])

# Найдём количество элементов в меньшем классе
minority_class = np.argmin(np.bincount(akb_labels))
minority_count = np.sum(akb_labels == minority_class)

# Индексы объектов каждого класса
indices_class_0 = np.where(akb_labels == 0)[0]
indices_class_1 = np.where(akb_labels == 1)[0]

# Случайно выбираем minority_count элементов из каждого класса
np.random.seed(42)  # для воспроизводимости
selected_indices_0 = np.random.choice(indices_class_0, minority_count, replace=False)
selected_indices_1 = np.random.choice(indices_class_1, minority_count, replace=False)

# Объединяем индексы
balanced_indices = np.concatenate([selected_indices_0, selected_indices_1])

# Получаем сбалансированные данные
akb_feat_bal = akb_features[balanced_indices]
akb_lab_bal = akb_labels[balanced_indices]

result = []
n_trees = 1000

for random_seed in tqdm(np.arange(1, 101)):
    iforest = IsolationForest(random_seed=random_seed, n_jobs=22, n_trees=n_trees).fit(features) # coniferst
    
    total_acc, total_nodes = get_forest_acc(iforest, akb_feat_bal, akb_lab_bal)
    acc, node, tree_id = get_tree_id(np.argmin(np.concat(total_acc)), total_acc, total_nodes)
    path_len = get_node_path(tree_id, node, iforest)
    
    scores = iforest.score_samples(features[:int(1e7)])
    arg_sort = np.argsort(scores)
    anom = features[arg_sort[:100]]
    
    anom_nodes = get_forest_nodes(iforest, anom)
    anom_fall = (anom_nodes[tree_id] == node).sum()
    
    result.append([random_seed, acc, node, tree_id, path_len, anom_fall])

result_csv = pd.DataFrame(data=result, columns=['random seed', 'Accuracy score', 'Node id', 'Tree id', 'Node depth', 'Top-100 anomaly count'])
result_csv.to_csv('../data/best_leaf_accuracy.csv', index=False)