import os
import numpy as np
import argparse
import onnxruntime as rt
import time
from tqdm import tqdm

def load_single(oid_filename, feature_filename):
    oid = np.memmap(oid_filename, mode='c', dtype=np.uint64)
    feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)
    return oid, feature

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

def load_rbmodel(model_name, num_threads=None):
    """Load ONNX model with memory optimization"""
    sess_options = rt.SessionOptions()
    sess_options.enable_mem_pattern = False  # Disable memory pattern for large models
    
    if num_threads:
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
    
    sess = rt.InferenceSession(
        f'{model_name}',
        sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    prob_name = sess.get_outputs()[1].name
    return sess, input_name, prob_name

def process_chunk(model, chunk_data, concat=False):
    """Process a single chunk of data"""
    sess, input_name, prob_name = model
    pred_proba = sess.run([prob_name], {input_name: chunk_data.astype(np.float32)})[0]
    proba = np.float32([pred[1] for pred in pred_proba])
    
    if concat:
        return np.hstack((chunk_data, proba.reshape((-1, 1))))
    return proba

def main():
    args = parse_arguments()
    
    # Load model
    model = load_rbmodel(args.modelname, num_threads=args.n_jobs)
    
    # Load data
    oids, features = load_single(f'{args.oid}', 
                                f'{args.feature}')
    #features = all_features[:, :-1]
    total_objects = len(oids)
    
    # Prepare output files
    output_file = f'{args.output}.dat'
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
        with open(f'{args.featurenames}') as f:
            names = f.read().split()
        exp_names = names + ['RB_clf_proba']
        with open(f'{args.output}.name', 'w') as f:
            for line in exp_names:
                f.write(f"{line}\n")
    
    total_time = (time.monotonic() - start_time) / 60
    print(f'\nCompleted processing {total_objects} objects in {total_time:.1f} minutes')

if __name__ == "__main__":
    main()
