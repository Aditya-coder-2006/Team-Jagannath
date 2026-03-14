import argparse
import csv
import sys

def validate_submission(csv_path: str):
    '''
    Performs sanity checks on submission.csv before submission.
    
    Checks:
    - correct CSV header (video_id,order)
    - no duplicate video IDs
    - no duplicate frame indices in a single row
    - frame indices are integers
    - indices start at 0
    - all indices from 0 to N-1 exist exactly once
    '''
    errors = []
    
    try:
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            if len(headers) != 2 or headers[0] != "video_id" or headers[1] != "order":
                errors.append(f"Header mismatch. Expected ['video_id', 'order'], got {headers}")
                
            seen_videos = set()
            for row_idx, row in enumerate(reader, start=2):
                if len(row) != 2:
                    errors.append(f"Row {row_idx}: Invalid number of columns. Expected 2, got {len(row)}.")
                    continue
                    
                video_id, order_str = row
                
                # Check duplicate video IDs
                if video_id in seen_videos:
                    errors.append(f"Row {row_idx}: Duplicate video_id found -> {video_id}")
                seen_videos.add(video_id)
                
                # Parse order string
                try:
                    order = [int(x) for x in order_str.strip().split()]
                except ValueError:
                    errors.append(f"Row {row_idx} ({video_id}): Non-integer values found in order sequence -> {order_str}")
                    continue
                
                N = len(order)
                if N == 0:
                    errors.append(f"Row {row_idx} ({video_id}): Order sequence is empty.")
                    continue
                
                # Convert to zero-based internally for validation
                zero_based = [x - 1 for x in order]
                # Check for duplicates within a row and exact 1 to N requirement
                expected_set = set(range(N))
                actual_set = set(zero_based)
                
                if len(zero_based) != len(actual_set):
                    errors.append(f"Row {row_idx} ({video_id}): Duplicate frame indices found in sequence.")
                
                if actual_set != expected_set:
                    missing = expected_set - actual_set
                    extra = actual_set - expected_set
                    msg = f"Row {row_idx} ({video_id}): Invalid sequence. "
                    if missing:
                        msg += f"Missing indices: {[m+1 for m in missing]}. "
                    if extra:
                        msg += f"Indices out of 1 to {N} bounds: {[e+1 for e in extra]}."
                    errors.append(msg.strip())

    except Exception as e:
        errors.append(f"Failed to read CSV file: {str(e)}")

    if errors:
        print("\n=== VALIDATION FAILED ===")
        print(f"Found {len(errors)} errors:")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        sys.exit(1)
    else:
        print("\n=== VALIDATION PASSED ===")
        print(f"Checked successfully. The submission format is completely correct and contains no duplicates.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission.csv format")
    parser.add_argument("--csv", type=str, default="submission.csv", help="Path to the submission CSV file")
    
    args = parser.parse_args()
    validate_submission(args.csv)