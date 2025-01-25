#!/bin/zsh


# Specify hyperparameters
hidden_dims=(128 1024 4542) 
batch_sizes=(16 64 256) 
learning_rates=(0.01 0.001 0.0005)  


# Check if sbatch is available
if command -v sbatch > /dev/null; then
  sbatch_cmd="sbatch"
else
  echo "\033[1;31mYou are not in a Slurm environment. Executing experiments sequentially!\033[0m"
  sbatch_cmd=""
fi



for hidden_dim in "${hidden_dims[@]}"; do
  for batch in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        if [ -z "$sbatch_cmd" ]; then
          echo "\033[1;32mExecuting job with environment variables:\033[0m -s $hidden_dim -b $batch -lr $lr"
          python predict_neighbour.py $hidden_dim $batch $lr
        else
          echo "hidden_dim = $hidden_dim __ batches = $batch __ learning rate = $lr"
          $sbatch_cmd --job-name="edge_ai-$hidden_dim-$batch-$lr" job.sh $hidden_dim $batch $lr
        fi
      done
    done
  done
done