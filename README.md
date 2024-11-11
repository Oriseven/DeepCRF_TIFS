# DeepCRF_TIFS

This repository provides the implementation of DeepCRF. Read the preprint for more details: TBD

## Setup

### Requirements
This repository is built on Python 3.9.18 and Pytorch 2.1.2.  
Other packages are listed in `requirements.txt`.

`pip3 install -r requirement.txt`

## Datasets:

Dataset could be downloaded from this file: [CSI Dataset](...).  
After downloading, please move it to the `data/` folder with the following directory structure:

```bash
$data/
  CSI/
  MAC/
  channel_BCDF.mat
  syn_testing_B_L.mat
  ...
  syn_testing_F.mat
```

## Training

```bash
# DeepCRF with contrastive loss
python3 main.py  Model='deepcrf-con' loss='contrastive'

# DeepCRF with ce loss
python3 main.py  Model='deepcrf' loss='cross'

# Baselines
python3 main.py  Model='ss' loss='cross'
python3 main.py  Model='self-acc' loss='cross'
python3 main.py  Model='att_network' loss='cross'
```

## Evaluation

```bash
# DeepCRF, evaluation with practical data
python3 evaluation.py  Model='deepcrf' test_with_practical_data=1 test_positions=['d3','d10','d12','d16','p1','p2','p3','outdoor','mobilenlos']

# DeepCRF, evaluation with synthetic data
python3 evaluation.py  Model='deepcrf' test_with_practical_data=0 channel_type=0 snr=40
```

## Citation

```
TBD
