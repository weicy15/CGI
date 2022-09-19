# Contrastive Graph Structure Learning via Information Bottleneck for Recommendation

This is the code in Contrastive Graph Structure Learning via Information Bottleneck for Recommendation which has been accepted by NeurIPS 2022.

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
```

## Data Process

To prepare the data for the model training:

```setup
python data_process.py
```

## Training

To train the model(s) in the paper:

```setup
python train.py
```
> Output: the file "model.tar"

## Evaluation

To evaluate my model in the paper:

```setup
python evaluate.py
```


