## Setup

To run our code, install:
```
conda create -n diffusent python=3.8
pip install -r requirements.txt
```

### Training

```
python diffusent.py train --config configs/14lap.conf
```

### Evaluating

Set the path of the model checkpoint into `eval.conf -> model_path`  and run:

```
python diffusent.py eval --config configs/eval.conf
```

