#!/bin/bash

module load miniconda

ENV_PATH="/projectnb/cs505am/students/$USER/envs/rag_env"

if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists."
else
    echo "Creating environment..."
    conda env create -p $ENV_PATH -f environment.yml
fi

conda activate $ENV_PATH

export HF_HOME=/projectnb/cs505am/materials/.cache/huggingface/

echo "HF_HOME set to $HF_HOME"

python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF

