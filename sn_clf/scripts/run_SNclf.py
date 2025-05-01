import os
import numpy as np
import argparse
import onnxruntime as rt
import time
from tqdm import tqdm

import sys
sys.path.insert(0, '../..')

from sn_clf.scripts.utils import load_features, load_snmodel, process_chunk

def make_argument_parser():
    parser = argparse.ArgumentParser(description='Real-bogus classification for ZTF objects')
    parser.add_argument('--oid', help='Name of the file with OIDs', required=True)
    parser.add_argument('--feature', help='Name of the file with corresponding features.', required=True)
    parser.add_argument('--featurenames', help='Name of the file with feature names.', required=True)
    parser.add_argument('--modelname', help='Trained model name.', required=True)
    parser.add_argument('--output', help='Output file name for results.', required=True)
    parser.add_argument('--concat', help='Concatenate probability to features', type=bool, default=False)
    parser.add_argument('--chunksize', help='Number of objects to process at once', type=int, default=10000000)
    parser.add_argument('--n_jobs', help='Number of threads to process the data', type=int, default=20)
    return parser

def parse_arguments():
    parser = make_argument_parser()
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    
    # Load model
    model = load_snmodel(args.modelname, num_threads=args.n_jobs)
    
    # Load data
    oids, features = load_features(f'../../dr23-features/{args.oid}', 
                                f'../../dr23-features/{args.feature}')
    total_objects = len(oids)
    
    # Prepare output files
    output_file = f'../../dr23-features/{args.output}.dat'
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Process in chunks
    print(f'Processing {total_objects} objects in chunks of {args.chunksize}...')
    start_time = time.monotonic()
    
    with open(output_file, 'ab') as f:  # Append mode for writing chunks
        for i in tqdm(range(0, total_objects, args.chunksize)):
            chunk_end = min(i + args.chunksize, total_objects)
            features_chunk = features[i:chunk_end]
            
            try:
                # Process chunk
                chunk_result = process_chunk(
                    model, 
                    features_chunk,
                    concat=args.concat
                )
                
                # Write chunk results
                f.write(chunk_result.tobytes())
                f.flush()  # Ensure data is written to disk
                
            except Exception as e:
                print(f"\nError processing chunk {i}-{chunk_end}: {str(e)}")
                # Try with smaller chunk size
                args.chunksize = max(1, args.chunksize // 2)
                print(f"Retrying with chunk size {args.chunksize}...")
                continue
    
    # Write feature names if concatenating
    if args.concat:
        with open(args.featurenames) as f:
            names = f.read().split()
        exp_names = names + ['SN_clf_proba']
        with open(f'../../dr23-features/{args.output}.name', 'w') as f:
            for line in exp_names:
                f.write(f"{line}\n")
    
    total_time = (time.monotonic() - start_time) / 60
    print(f'\nCompleted processing {total_objects} objects in {total_time:.1f} minutes')

if __name__ == "__main__":
    main()
