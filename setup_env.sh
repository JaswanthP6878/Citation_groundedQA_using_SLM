#!/bin/bash

module load miniconda

ENV_NAME="${ENV_NAME:-rag_env}"

DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/zukhriddin_nlp_research_project"
if [[ ! -d "$DEFAULT_PROJECT_ROOT" && -d "/projectnb/cs505am/students/$USER/nlp_research_project" ]]; then
    DEFAULT_PROJECT_ROOT="/projectnb/cs505am/students/$USER/nlp_research_project"
fi

if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment already exists."
else
    echo "Creating environment..."
    conda env create -n $ENV_NAME -f environment.yml
fi

conda activate $ENV_NAME

export HF_HOME="${HF_HOME:-/projectnb/cs505am/students/$USER/.cache/huggingface/}"
mkdir -p "$HF_HOME"

echo "Project root default: ${NLP_PROJECT_USER_ROOT:-$DEFAULT_PROJECT_ROOT}"
echo "HF_HOME set to $HF_HOME"

python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF
