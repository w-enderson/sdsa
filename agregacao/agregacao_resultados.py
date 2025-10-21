import pandas as pd
import os
import re

def combine_results():
    # Path to the results directory
    results_path = "../results_test"
    
    # Regex pattern to match filenames like: dataset-<anything>-mc-<number>.csv
    pattern = re.compile(r'^dataset-.*-mc-\d+\.csv$', re.IGNORECASE)

    # List to store all dataframes
    all_dfs = []
    processed = 0
    ignored = 0
    errors = 0

    # Check if directory exists
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory '{results_path}' not found")
    
    # Loop through all subdirectories in results_test
    for subdir in os.listdir(results_path):
        subdir_path = os.path.join(results_path, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                # Only consider CSV files that match the pattern
                if file.lower().endswith('.csv') and pattern.match(file):
                    file_path = os.path.join(subdir_path, file)
                    try:
                        # Read each CSV file
                        df = pd.read_csv(file_path)
                        # Optionally add filename and directory name as columns
                        df['source_file'] = file
                        df['algorithm'] = subdir
                        all_dfs.append(df)
                        processed += 1
                    except Exception as e:
                        errors += 1
                        print(f"Error reading file {file}: {str(e)}")
                else:
                    # Ignore files that do not match the dataset-...-mc-<n>.csv pattern
                    ignored += 1

    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined results
        output_path = "combined_results.csv"
        combined_df.to_csv(output_path, index=False, float_format="%.6f")
        print(f"Combined results saved to {output_path}")
        print(f"Processed: {processed}, Ignored: {ignored}, Errors: {errors}")
        return combined_df
    else:
        print("No CSV files found in the results directory matching the required pattern")
        print(f"Processed: {processed}, Ignored: {ignored}, Errors: {errors}")
        return None

def calculate_statistics(df):
    """Calculate mean and variance of accuracy and execution time by dataset and classifier"""
    
    # Group by dataset and classifier
    stats = df.groupby(['dataset', 'classifier']).agg({
        'acc': ['mean', 'std'],
        'exec_time': ['mean', 'std']
    })
    
    # Rename columns for better readability
    stats.columns = [
        'acc_mean', 'acc_std',
        'time_mean', 'time_std',
    ]
    stats = stats.reset_index()
    
    # Round numerical values to 6 decimal places
    stats = stats.round(6)
    
    # Save statistics to CSV
    output_file = 'statistics_results.csv'
    stats.to_csv(output_file, index=False, float_format="%.6f")
    print(f"\nStatistics saved to {output_file}")
    return stats

if __name__ == "__main__":
    combined_results = combine_results()
    if combined_results is not None:
        statistics = calculate_statistics(combined_results)
