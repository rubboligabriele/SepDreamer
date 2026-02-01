# MedDreamer

This repository contains an implementation of MedDreamer framework, tailored for sequential decision-making tasks in Electronic Health Records (EHR) data. 

## Prerequisites

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Run the Code

Default is world model training, if want to train policy, modify the command as p1/p2-sepsis/vent

```bash
python main.py --config p1-sepsis
```

Monitor results:

```bash
tensorboard --logdir ./run_log
```

## Acknowledgement
The code repository is based on: 
- NM512's pytorch implementation of dreamerv3: https://github.com/NM512/dreamerv3-torch/tree/main
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
