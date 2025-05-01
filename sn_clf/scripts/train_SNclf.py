import time
import sys
import argparse
import requests
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import os
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn import metrics
import pandas as pd

import sys
sys.path.insert(0, '../..')

from sn_clf.scripts.utils import get_sn_label_from_akb, load_features, download_akb_json, convert_to_onnx


def make_argument_parser():
    parser = argparse.ArgumentParser(description='Train SN / non-SN classification model for ZTF objects')
    parser.add_argument('--featurenames', help='Name of the file with feature names, one name per line', required=True)
    parser.add_argument('--featuresfile', help='Name of the file with features', required=True)
    parser.add_argument('--oidsfile', help='Name of the file with OIDs', required=True)
    parser.add_argument('--modelname', help='Name for trained model.', required=True)
    parser.add_argument('--akb_crossmatch', help='Crossmatch SNe from akb with oidsfile or not.', default=False)
    parser.add_argument('--bts_crossmatch', help='Crossmatch SNe from bts with oidsfile or not.', default=False)
    parser.add_argument('--akb_objects', help='Name of akb database file.', default=False)
    parser.add_argument('-s', '--random_seed', default=42, type=int, help='Fix the seed for reproducibility. Defaults set to 42.')
    return parser

def parse_arguments():
    parser = make_argument_parser()
    args = parser.parse_args()
    return args
    

def main():
    args = parse_arguments()
    np.random.seed(args.random_seed)
    print('Start..')
    if not args.akb_objects:
        args.akb_objects = 'akb_objects.json'
        download_akb_json('akb_objects.json')
    
    akb_oids, akb_sn_label = get_sn_label_from_akb(f'../data/{args.akb_objects}')
    
    with open(f'../../dr23-features/{args.featurenames}') as f:
        names = f.read().split()


    oids, features = load_features(f'../../dr23-features/{args.oidsfile}', f'../../dr23-features/{args.featuresfile}')

    if args.akb_crossmatch:
        akb_crossmatch = np.load(f'../data/{args.akb_crossmatch}')
    else:
        t = time.monotonic()
        akb_oid_sn = akb_oids[akb_sn_label == 1]
        akb_crossmatch = np.isin(oids, akb_oid_sn)
        print(f'Crossmatched in {np.round((time.monotonic() - t) / 60)} min')
        np.save(f'../data/akb_sn_dr23_crossmatch.npy', akb_crossmatch)


    bts_sn = pd.read_csv('../data/bts_crossmatched_2sec.csv')
    if args.bts_crossmatch:
        bts_crossmatch = np.load(f'../data/{args.bts_crossmatch}')
    else:
        bts_oids = list(bts_sn['OID'])
        bts_crossmatch = np.isin(oids, bts_oids)
        print(f'Crossmatched in {np.round((time.monotonic() - t) / 60)} min')
        np.save(f'../data/bts_dr23_crossmatch.npy', bts_crossmatch)


    # тут в качестве негативного класса используются рандомные объекты из дата релиза
    bts_features = features[bts_crossmatch] # содержит только SN
    akb_sn_features = features[akb_crossmatch]

    indx = np.random.choice(np.arange(len(oids)), len(bts_features)+len(akb_sn_features))
    regular_obj = features[indx]
    train_data = np.vstack([bts_features, regular_obj[:len(bts_features)]])
    test_data = np.vstack([akb_sn_features, regular_obj[len(bts_features):]])

    train_labels = np.hstack([np.ones(len(bts_features)), np.zeros(len(bts_features))])
    test_labels = np.hstack([np.ones(len(akb_sn_features)), np.zeros(len(akb_sn_features))])


    # Train and validate real-bogus model
    print('Training model...')
    t = time.monotonic()
    model = RandomForestClassifier(max_depth=18, n_estimators=831, random_state=args.random_seed)
    score_types = ('accuracy', 'roc_auc', 'f1')

    result = cross_validate(model, train_data, train_labels,
                        cv=KFold(shuffle=True, random_state=args.random_seed),
                        scoring=score_types,
                        return_estimator=True,
                        return_train_score=True,
                        n_jobs=5,
                       )

    print('Scores for Random Forest Classifier:')
    for score in score_types:
        mean = np.mean(result[f'test_{score}'])
        std = np.std(result[f'test_{score}'])
        print(f'{score} = {mean:.3f} +- {std:.3f}')
    t = (time.monotonic() - t) / 60
    print(f'RF trained (with cross-validation) in {t:.0f} m')
    
    assert np.mean(result['test_accuracy']) > 0.7, 'Accuracy for trained model is too low!'
    clf = result['estimator'][0]

    convert_to_onnx(clf, len(names), name=args.modelname)

    test_pred = clf.predict_proba(test_data)
    akb_roc_auc = metrics.roc_auc_score(test_labels, test_pred[:, 1])
    akb_accuracy = clf.score(test_data, test_labels)
    print('Results on test data (SN from akb):')
    print(f'ROC-AUC = {akb_roc_auc:.3f}')
    print(f'Accuracy = {akb_accuracy:.3f}')

if __name__ == "__main__":
    main()