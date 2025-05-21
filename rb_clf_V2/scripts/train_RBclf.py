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




def make_argument_parser():
    parser = argparse.ArgumentParser(description='Train real-bogus classification model for ZTF objects')
    parser.add_argument('--featurenames', help='Name of the file with feature names, one name per line', required=True)
    parser.add_argument('--featuresfile', help='Name of the file with features', required=True)
    parser.add_argument('--oidsfile', help='Name of the file with OIDs', required=True)
    parser.add_argument('--modelname', help='Name for trained model.', required=True)
    parser.add_argument('--crossmatch', help='Crossmatch objects from akb with oidsfile or not.', default=False)
    parser.add_argument('--akbfeat', help='Name for saving AKB object features.')
    parser.add_argument('--akblist', help='Name of akb database file.', default=False)
    parser.add_argument('-s', '--random_seed', default=42, type=int, help='Fix the seed for reproducibility. Defaults set to 42.')
    return parser

def parse_arguments():
    parser = make_argument_parser()
    args = parser.parse_args()
    return args


def load_single(oid_filename, feature_filename):
    oid     = np.memmap(oid_filename, mode='c', dtype=np.uint64)
    feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)
    return oid, feature


def convert_to_onnx(model, input_shape, name):    
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(f'../models/RBclf_{name}.onnx', "wb") as f:
        f.write(onx.SerializeToString())


def get_akb_labels(filepath):
    file = open(filepath)
    obj_list = json.load(file)
    file.close()

    oids = []
    tags = []
    for data in obj_list:
        oids.append(data['oid'])
        tags.append(data['tags'])

    # 0-artefact,  1-transient
    result = {}
    for oid, tag_list in zip(oids, tags):
        tag_list = set(tag_list)
        if tag_list == set(['Galaxy']):
            continue
        if tag_list == set(['Galaxy', 'uncertain']):
            continue
        if tag_list == set(['interaction']):
            continue
        if tag_list == set(['1-point']):
            continue
        if tag_list == set(['LSB', 'uncertain']):
            continue
        if tag_list == set(['M_dwarf_flare', 'uncertain']):
            continue
        if tag_list == set(['VAR', 'uncertain']):
            continue
        if 'STAR' in tag_list:
            continue
        if 'Asteroid' in tag_list:
            continue
        if 'Comet' in tag_list:
            continue
        if 'artefact' in tag_list and 'uncertain' in tag_list:
            continue

        
        if 'artefact' in tag_list:
            result[oid] = 0
        
        else:
            result[oid] = 1
    
    return result


def main():
    args = parse_arguments()
    print('Start..')
    if not args.akblist:
        args.akblist = 'akb.ztf.snad.space.json'
        url = f'https://akb.ztf.snad.space/objects/'
        with requests.get(url) as response:
            response.raise_for_status()
            open(f'../data/{args.akblist}', 'wb').write(response.content)
    
    akb = get_akb_labels(f'../data/{args.akblist}')
    with open(f'../../dr23-features/{args.featurenames}') as f:
        names = f.read().split()


    oids, features = load_single(f'../../dr23-features/{args.oidsfile}', f'../../dr23-features/{args.featuresfile}')

    if args.crossmatch:
        crossmatch = np.load(f'../data/{args.crossmatch}')
    else:
        t = time.monotonic()
        crossmatch = np.isin(oids, list(akb.keys()))
        print(f'Crossmatched in {np.round((time.monotonic() - t) / 60)} min')
        np.save(f'../data/crossmatch.npy', crossmatch)
        

    akb_features = features[crossmatch]
    akb_labels = np.array([akb[oid] for oid in oids[crossmatch]]).reshape((-1,1))
    data = pd.DataFrame(data=np.hstack([akb_features, akb_labels]), columns=names+['label'])


    # Train and validate real-bogus model
    print('Training model...')
    t = time.monotonic()
    model = RandomForestClassifier(max_depth=18, n_estimators=831, random_state=args.random_seed)
    score_types = ('accuracy', 'roc_auc', 'f1')

    result = cross_validate(model, data[names], data['label'],
                        cv=KFold(shuffle=True, random_state=args.random_seed),
                        scoring=score_types,
                        return_estimator=True,
                        return_train_score=True,
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


if __name__ == "__main__":
    main()
