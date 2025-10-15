import pandas as pd
import os

def combine_results():
    # Path to the results directory
    results_path = "../results_test"
    
    # List to store all dataframes
    all_dfs = []
    
    # Check if directory exists
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Directory '{results_path}' not found")
    
    # Loop through all subdirectories in results_test
    for subdir in os.listdir(results_path):
        subdir_path = os.path.join(results_path, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir_path, file)
                    try:
                        # Read each CSV file
                        df = pd.read_csv(file_path)
                        # Add filename and directory name as columns if needed
                        # df['source_file'] = file
                        # df['algorithm'] = subdir
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading file {file}: {str(e)}")
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined results
        output_path = "combined_results.csv"
        combined_df.to_csv(output_path, index=False, float_format="%.6f")
        print(f"Combined results saved to {output_path}")
        return combined_df
    else:
        print("No CSV files found in the results directory")
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
