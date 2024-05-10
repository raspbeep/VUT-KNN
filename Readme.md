# Sketch-to-Face Translation Using Cycle-Consistent GAN 
### Brno University of Technology, FIT, Course: KNN
In this project, we utilized the base CGAN solution proposed by Jun-Yan et al. [1]. Additionally, we improved the baseline solution by incorporating the Self-Attention mechanism [2] and the Identity Loss function [3].

## Team 
**Pavel Kratochvil** (xkrato61)  
**Lucie Svobodova** (xsvobo1x)  
**Jakub Kuznik** (xkuzni04)  

## Datasets 
| Dataset  | Description |
|----------|--------------|
| [URLSketch2face](https://vutbr-my.sharepoint.com/:f:/g/personal/xsvobo1x_vutbr_cz/Eo0xnlq_sQdJle7S9pc3xBYB0pSw-w5M_i5VM7favWMc_A?e=t5edPI)  | Contains face sketches and images. |     
| [Horse2zebra](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset) | Contains images of horses and zebras. | 

## Purpose of Files  
| File        | Purpose |
|-------------|--------------------------------------------------------------|
| config.py   | Configures various parameters and settings essential for training |        
| dataset.py   | Prepare dataset for training/evaluation. |        
| discriminator.py | Contains discriminator model definition |        
| generator_model.py | Contains generator model definition |        
| image_pool.py | Definition of the image pool. |        
| utils.py | Handle utility functions for saving and loading the trained model. |  
| doc/* |  Documentation of the project. | 

## Installation dependencies 
```pip install -r requirements.txt``` 

### Run training
```python train.py```

### Run evaluation
```python3 forwardpass.py```

## Saved models  
Saved models are aviable at: [models](https://vutbr-my.sharepoint.com/:f:/g/personal/xsvobo1x_vutbr_cz/EmT0auMzEz5HvfKHwumq6ssBxui7uz73oK0EKuHFd3byIg?e=nZ5wAh)

## Sources 

[1] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." arXiv preprint arXiv:1703.10593 [cs.CV] (2017).

[2] Zhang, Han & Goodfellow, Ian & Metaxas, Dimitris & Odena, Augustus. (2018). Self-Attention Generative Adversarial Networks. 

[3] Liu, S. (2022). Study for Identity Losses in Image-to-Image Domain Translation with Cycle-Consistent Generative Adversarial Network. Journal of Physics: Conference Series, 2400(1), 012030. doi: 10.1088/1742-6596/2400/1/012030

[4] [CGAN implementation](https://www.youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va)