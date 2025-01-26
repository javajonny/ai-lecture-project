#!/bin/zsh

# Define the values for each hyperparameter
hidden_dims=(128 1024 4542) 
batch_sizes=(512 1024 2048) 
learning_rates=(0.01 0.001 0.0005) 

# Path to the Python script
python_script="/Users/jonas/Programming/ai-lecture-project/predict_neighbours.py"

# Create a directory to save the logs
log_dir="hyperparameter_logs"
mkdir -p $log_dir

# Iterate over all combinations of hyperparameters
for hidden_dim in "${hidden_dims[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "Running with hidden_dim=$hidden_dim, learning_rate=$learning_rate, batch_size=$batch_size"
            
            # Generate a unique log file name
            log_file="$log_dir/hidden${hidden_dim}_lr${learning_rate}_batch${batch_size}.log"

            # Execute the Python script and save the output to the log file
            python $python_script $hidden_dim $batch_size $learning_rate > $log_file 2>&1

            echo "Finished: hidden_dim=$hidden_dim, learning_rate=$learning_rate, batch_size=$batch_size"
            echo "Log saved to $log_file"
        done
    done
done

echo "Hyperparameter search completed. Logs saved to $log_dir"
