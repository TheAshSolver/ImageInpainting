import argparse
import os
from cleanfid import fid

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a set of files using the FID score.")
    parser.add_argument(
        "--real_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing reference/real image files."
    )
    parser.add_argument(
        "--gen_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing generated/candidate image files."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Computation device to use, e.g., 'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="v3_inception", 
        help="Feature extractor model variant (default: 'v3_inception')."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Validate directories
    if not os.path.isdir(args.real_dir):
        raise NotADirectoryError(f"Reference directory not found: {args.real_dir}")
    if not os.path.isdir(args.gen_dir):
        raise NotADirectoryError(f"Candidate directory not found: {args.gen_dir}")

    print("=" * 50)
    print("Running FID Benchmark")
    print(f"Reference (Real) Path : {args.real_dir}")
    print(f"Candidate (Gen) Path  : {args.gen_dir}")
    print(f"Device                : {args.device}")
    print("=" * 50)

    # Compute FID score
    score = fid.compute_fid(
        args.real_dir,
        args.gen_dir,
        device=args.device,
        model_name=args.model_name,
        use_dataparallel=False    args.real_dir,
        args.gen_dir,
        device=args.device,
        model_name=args.model_name,
        use_dataparallel=False 
    )

    print("\n" + "=" * 50)
    print(f"Final FID Score: {score:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()