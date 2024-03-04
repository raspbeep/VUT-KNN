# KNN 
Aim of this project is to use CGAN (Conditional Generative Advesarial Network) for (topic tbd) 

## Team 
**Pavel Kratochvil** (xkrato61)  
**Lucie Svobodova** (xsvobo1x)  
**Jakub Kuznik** (xkuzni04)  


## Datasets 
| Dataset  | URL | Description |
|----------|-----|-------------|
| tbd | tbd | tbd |     

## Purpose of Files  
| File        | Purpose |
|-------------|--------------------------------------------------------------|
| config.py   | Configures various parameters and settings essential for training |        
| dataset.py   | Prepare dataset for training/evaluation. |        
| discriminator.py | Contains discriminator model definition |        
| generator_model.py | Contains generator model definition |        
| utils.py | Handle utility functions for saving and loading the trained model. |  
| data/train/* |  Training data | 
| data/val/*   | Validation data |  

## Evaluation 
tbd


## Installation Guide
### Run training
```python train.py```
### Run Evaluation
tbd
### Use model  
tbd

### Setting up Python Environment

- Create a Python virtual environment in the current folder:  
   ```python -m venv .``` 
- Activate the venv:  
   ```  source bin/activate```   
- Install the required packate into the venv:  
   ```pip3 install package```  
- Deactivate venv:   
   ```deactivate```  
- Create requirements.txt file:   
   ```python -m pip freeze > requirements.txt``` 
- Install from requirements  
    ```pip install -r requirements.txt``` 


## Sources 

CGAN: 
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." arXiv preprint arXiv:1703.10593 [cs.CV] (2017).

CGAN implementation: 
https://www.youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va