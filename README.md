# Revisiting Pretraining Objectives for Tabular Deep Learning
This is the official code for our paper "Revisiting Pretraining Objectives for Tabular Deep Learning"

**Check out other projects on tabular Deep Learning:** [link](https://github.com/Yura52/rtdl#papers-and-projects).

Feel free to report [issues](https://github.com/puhsu/tabular-dl-pretrain-objectives/issues) and post [questions/feedback/ideas](https://github.com/puhsu/tabular-dl-pretrain-objectives/discussions).

## Results
You can view all the results and build your own tables with this [notebook](bin/Reports.ipynb).

## Setup the environment
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (just to manage the env).
2. Run the following commands
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n tdl python=3.9.7
    conda activate tdl

    pip install torch==1.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    # if the following commands do not succeed, update conda
    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}
    
    conda activate tdl
    ```

## Running the experiments

Here we describe the neccesary info for reproducing the experimental results.

### Datasets

We upload the datasets used in the paper with our train/val/test splits [here](https://www.dropbox.com/s/cj9ex11u6ri0tdy/tabular-pretrains-data.tar?dl=1). We do not impose additional restrictions to the original dataset licenses, the sources of the data are listed in the paper appendix.

You could load the datasets with the following commands:

``` bash
conda activate tdl
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/cj9ex11u6ri0tdy/tabular-pretrains-data.tar?dl=1" -O tabular-pretrains-data.tar
tar -xvf tabular-pretrains-data.tar
```

### File structure

There are multiple scripts inside the `bin` directory for various pretraining objectives, finetuning from checkpoints (same script is also used to train from scratch) and GBDT baselines.

Each pretraining script follows the same structure. It constructs different models given their configs (MLPs, MLPs with numerical embeddings, ResNets, Transformers) and pretrains them with periodically calling the finetune script for early stopping (or finetuning only at the end if `early_stop_type = "pretrain"` is specified in config).

There are two variations of each script: single GPU and DDP multi-GPU (used for large dataset and models with embeddings), which are identical, except DDP related modifications. 

- `bin/finetune.py` are used to train models from scratch, or finetune pretrained checkpoints
- `bin/contrastive.py` -- contrastive objective.
- `bin/[rec|mask]_(supervised)` -- self-prediction objective variations

### Example
To run the target-aware mask prediction pretraining on the california housing dataset you could run the following code snippet. It will clone the tuning config, then tune and evaluate mlp-plr with target-aware mask prediction pretraining and create the ensemble

``` bash
conda activate tdl
cd $PROJECT_DIR
mkdir -p exp/draft
cp exp/mask-target/mlp-p-lr/california/3_tuning.toml exp/draft/example_tuning.toml

export CUDA_VISIBLE_DEVICES=0
python bin/tune.py exp/draft/example_tuning.toml
python bin/evaluate.py exp/draft/example_tuning 15
python bin/ensemble.py exp/draft/example_evaluation
```

