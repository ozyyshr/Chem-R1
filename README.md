## Quick start

(1) Data preparation for Chemistry QA.
```bash
#!/bin/bash

# Define an array with the script paths
scripts=(
    "scripts/data_process/chemcot.py"
    "scripts/data_process/scibench.py"
    "scripts/data_process/mmlu_chem.py"
    "scripts/data_process/moleculeqa.py"
)

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python "$script"
    if [ $? -ne 0 ]; then
        echo "Error: $script failed to execute."
        exit 1
    fi
    echo "$script completed successfully."
done
```