# ConsistencyModelImplementation

I tried implementing the Consistency distillation from the paper https://arxiv.org/pdf/2303.01469.pdf

The model is from https://github.com/NVlabs/edm and I used some code from https://github.com/openai/consistency_models


To train the model use the following command `python ./Code/consistency.py --teacher_model_dir= --data_dir="./data" --save_dir="./model"
The pre trained model is found here https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl
