import argparse
from tqdm import tqdm
from os import path as osp
from datasets import build_dataset
from utils.options import parse

def get_worker_assignment(total_size, worker_id, num_workers):
    """
    Distribute items as evenly as possible among workers.
    When total_size isn't evenly divisible, spread the remainder
    across the first R workers, where R is the remainder.
    """
    base_size = total_size // num_workers
    remainder = total_size % num_workers
    
    # Workers with ID < remainder get one extra item
    if worker_id < remainder:
        start_idx = worker_id * (base_size + 1)
        items = base_size + 1
    else:
        start_idx = (worker_id * base_size) + remainder
        items = base_size
        
    end_idx = start_idx + items
    return start_idx, end_idx

def preprocess_dataset(root_path, worker_id, num_workers):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0 to num_workers-1)')
    parser.add_argument('--num_workers', type=int, required=True, help='Total number of workers')
    args = parser.parse_args()
    
    # Validate worker arguments
    if args.worker_id >= args.num_workers:
        raise ValueError(f"Worker ID ({args.worker_id}) must be less than number of workers ({args.num_workers})")
    if args.worker_id < 0 or args.num_workers < 1:
        raise ValueError("Worker ID must be non-negative and number of workers must be positive")
    
    opt = parse(args.opt, root_path, is_train=False)
    
    # Process each dataset in the config
    for dataset_name, dataset_opt in opt["datasets"].items():
        if isinstance(dataset_opt, int):  # skip batch_size, num_worker entries
            continue
            
        # Print dataset config (only for first worker to avoid spam)
        if args.worker_id == 0:
            print("\n" + "="*50)
            print(f"Dataset: {dataset_name}")
            for key, value in dataset_opt.items():
                if isinstance(value, (str, int, bool)):
                    print(f"{key}: {value}")
            print("="*50 + "\n")
        
        # Build dataset
        if args.worker_id == 0:
            print(f"Building dataset {dataset_name}...")
        test_set = build_dataset(dataset_opt)
        total_size = len(test_set)
        if args.worker_id == 0:
            print(f"Total dataset size: {total_size}")
        
        # Calculate this worker's portion of the dataset
        start_idx, end_idx = get_worker_assignment(total_size, args.worker_id, args.num_workers)
        
        # Iterate through this worker's portion of the dataset
        print(f"Worker {args.worker_id}: Processing items {start_idx} to {end_idx-1} ({end_idx - start_idx} items)")
        for i in tqdm(range(start_idx, end_idx), 
                     desc=f"Worker {args.worker_id}",
                     position=args.worker_id):
            data = test_set[i]
            # Processing happens in the dataset's __getitem__

if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0 to num_workers-1)')
    parser.add_argument('--num_workers', type=int, required=True, help='Total number of workers')
    args, _ = parser.parse_known_args()  # Use known_args to handle --opt separately
    preprocess_dataset(root_path, args.worker_id, args.num_workers) 