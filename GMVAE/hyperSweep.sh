#!/bin/bash

# -----------------------------
# Hyperparameters to sweep
# -----------------------------
learning_rates=(1e-4)
hidden_dims=( 1024 )
numbers=(10)

# -----------------------------
# Max parallel jobs
# Set to number of CPU cores or GPU processes
# -----------------------------
MAX_JOBS=4

# -----------------------------
# Helper function to limit parallel jobs
# -----------------------------
wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# -----------------------------
# Run all combinations
# -----------------------------
for lr in "${learning_rates[@]}"; do
    for hd in "${hidden_dims[@]}"; do
        for num in "${numbers[@]}"; do
            echo "Launching run: lr=$lr  hidden_dim=$hd  numbers=$num"

            # Run Python training script in background
            python /zhome/7f/5/168608/CCVAE/GMVAE/CCVAE_new_script.py "$lr" "$hd" "$num" 3000 > "/zhome/7f/5/168608/CCVAE/GMVAE/logs/run_lr${lr}_hd${hd}_num${num}.log" 2>&1 &

            # Enforce parallel job limit
            wait_for_jobs
        done
    done
done

# Wait for all children to finish
wait

echo "All sweeps completed."
