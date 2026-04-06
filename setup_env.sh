#!/bin/bash

module load miniconda

ENV_NAME="rag_env"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment already exists."
else
    echo "Creating environment..."
    conda env create -n $ENV_NAME -f environment.yml
fi

conda activate $ENV_NAME

export HF_HOME=/projectnb/cs505am/materials/.cache/huggingface/

echo "HF_HOME set to $HF_HOME"

python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF

