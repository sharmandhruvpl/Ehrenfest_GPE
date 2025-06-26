#!/bin/bash
# batch_submit.sh - Run jobs in batches of 5

# Create directories
mkdir -p quantum_dynamics_results
mkdir -p logs
mkdir -p job_scripts

# Configuration
PARTITION="batch"    # Use the correct partition
MEM="16G"
TIME="04:00:00"
BATCH_SIZE=5

# Generate parameter files
echo "Generating parameter files..."
python create_params.py

# Get all parameter files
param_files=(param_files/*.json)
total_jobs=${#param_files[@]}
echo "Found $total_jobs parameter files"

# Function to submit a batch of jobs and return job IDs
submit_batch() {
    local start=$1
    local end=$2
    local job_ids=()
    
    echo "Submitting batch: jobs $start to $end"
    
    for ((i=start; i<=end && i<total_jobs; i++)); do
        param_file="${param_files[$i]}"
        job_name=$(basename "$param_file" .json)
        
        # Create job script - WITHOUT module commands
        cat > "job_scripts/${job_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=logs/${job_name}_%j.log
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --partition=$PARTITION

echo "Starting job at \$(date)"
echo "Running on node: \$(hostname)"
echo "Python version: \$(python --version 2>&1)"

# Run simulation directly without loading modules
python hpc_oehrenn.py --run --param-file $param_file --output-dir quantum_dynamics_results --run-id "${job_name}"

echo "Job completed at \$(date)"
EOF
        
        # Submit job and store ID
        job_id=$(sbatch "job_scripts/${job_name}.sh" | awk '{print $4}')
        job_ids+=($job_id)
        
        echo "Submitted job $i: $job_name (ID: $job_id)"
    done
    
    # Return comma-separated list of job IDs
    echo "${job_ids[@]}" | tr ' ' ','
}

# Function to check if any jobs in a batch are still running
jobs_running() {
    local job_list=$1
    
    for job_id in $(echo $job_list | tr ',' ' '); do
        if squeue -j $job_id 2>/dev/null | grep -q $job_id; then
            return 0  # At least one job is still running
        fi
    done
    
    return 1  # No jobs from this batch are running
}

# Submit jobs in batches of 5 and wait for each batch to complete
current_batch=0
total_batches=$(( (total_jobs + BATCH_SIZE - 1) / BATCH_SIZE ))

while ((current_batch < total_batches)); do
    start_idx=$((current_batch * BATCH_SIZE))
    end_idx=$((start_idx + BATCH_SIZE - 1))
    
    echo "========================================"
    echo "Submitting batch $((current_batch + 1)) of $total_batches"
    echo "========================================"
    
    # Submit this batch and get job IDs
    job_ids=$(submit_batch $start_idx $end_idx)
    
    # Wait for all jobs in this batch to complete
    echo "Waiting for batch $((current_batch + 1)) to complete..."
    while jobs_running "$job_ids"; do
        echo "[$(date)] Batch $((current_batch + 1)) still running, checking again in 60 seconds..."
        sleep 60
    done
    
    echo "Batch $((current_batch + 1)) completed!"
    current_batch=$((current_batch + 1))
done

echo "All $total_jobs jobs have been processed!"
