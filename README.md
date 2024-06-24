# GADIFF: A Transferable Graph Attention Diffusion Model for Generating Molecular Conformations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/GeoDiff/blob/main/LICENSE)

## Environments

### Install via Conda

```bash
# Clone the environment
conda env create -f environment.yml
# Activate the environment
conda activate gadiff
```

## Dataset

### Offical Dataset
This is the [[offical raw GEOM dataset]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

### Preprocessed dataset
We use the preprocessed datasets (GEOM-QM9 and GEOM-Drugs) of GeoDiff, which can be obtained through [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing). The detailed usage are illustrated in [[GeoDiff codebase]](https://github.com/MinkaiXu/GeoDiff).

## Training

The hyper-parameters are in config files (`./configs/*.yml`). The following commands are used for training the model:

```bash
# Default settings
python train.py ./config/qm9_para.yml
python train.py ./config/drugs_para.yml
```

The model checkpoints, configuration yaml files, training logs will all be saved into a directory specified by `--logdir` in `train.py`.

## Generation

The checkpoints of two trained models of GEOM Datasets, i.e., `QM9` and `Drugs` is in the [[google drive folder]](https://drive.google.com/drive/folders/1sCS89cpbtCBDaFLiggKrCmyFFIFW2Beo?usp=drive_link). Then, please put the checkpoints `*.pt` into paths like `${log}/${model}/checkpoints/`, and also put related configuration file `*.yml` into the upper level directory `${log}/${model}/`.

You can generate conformations for test sets by:

```bash
python test.py ${log}/${model}/checkpoints/${iter}.pt \
    --start_idx 800 --end_idx 1000
```
Here `start_idx` and `end_idx` indicate the range of the test set to be generated. All sampling hyper-parameters can be set in `test.py` files.

## Evaluation

After generating conformations, the two tasks are utilized to evaluation the conformation quality.

### Task 1. Conformation Generation

The `COV` and `MAT` scores on the generated conformations can be computed using the following commands:

```bash
python eval_covmat.py ${log}/${model}/${sample}/sample_all.pkl
```

### Task 2. Property Prediction

For the property prediction, the small qm9 split in the [[google drive folder]](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing) is used for generating conformations with following commands:

```bash
python ${log}/${model}/checkpoints/${iter}.pt --num_confs 50 \
      --start_idx 0 --test_set data/GEOM/QM9/qm9_property.pkl
```

Generating conformations are evaluate `mean absolute errors (MAE)` metric on this split using the following commands:

```bash
python eval_prop.py --generated ${log}/${model}/${sample}/sample_all.pkl
```

## Acknowledgement

This repo is built upon the diffusion framework of excellent work, [[GeoDiff]](https://github.com/MinkaiXu/GeoDiff).

## Contact

If you have any question, please contact at wangdh608@nenu.edu.cn.


