# NLP_research_project

## Reports

LaTeX source and build instructions for all CS505 milestone reports (Proposal, Midway, Final) are in the [`reports/`](reports/README.md) directory.

## SCC setup 

create .condarc file in $HOME
add the following there
```
envs_dirs:
  - /projectnb/cs505am/students/$USER/.conda/envs
  - ~/.conda/envs

pkgs_dirs:
  - /projectnb/cs505am/students/$USER/.conda/pkgs
  - ~/.conda/pkgs

env_prompt: ({name})
```
this is to create a conda env within our respective folders in the scc

clone the project into your $USER folder and run the `./setup_env.sh` (if the base file is not running use `chmod +x setup_env.sh`)

The conda env used for this project is named as `rag_env`
