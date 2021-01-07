# dml-tf
distance metric learning, tf2 implementation

## How To Use
Before you train, configure train/config.py.
```
git clone https://github.com/shi510/dml-tf
cd dml-tf
export PYTHONPATH=$PYTHONPATH:$(pwd)
python train/main.py
```

## Training Conditions
1. Augmentation = random_flip, random_color, random_cutout  
2. Random Crop = All images are resized by 256x256 and then random crop by 224x224.  
3. Proxy Scale Factor = 8, `this is important. (Depending on your dataset size)`  
4. All network should be pretrained by imagenet, `this is important.`  
   - Issue: MobileNet not converges.  

## Results of ProxyNCA On Test Set
|                   | CUB         | cars196     |
|-------------------|-------------|-------------|
| NMI               | 73%         | 73%         |
| Linear Eval. Acc. | 68%         | 75%         |
| Epoch             | 15          | 15          |
| Arch.             | InceptionV3 | InceptionV3 |
| Batch Size        | 64          | 64          |
| Embedding Size    | 64          | 64          |
| Proxy Scale       | 8           | 8           |
| Embedding Scale   | 1           | 1           |
| Optimizer         | Adam@1e-4   | Adam@1e-4   |
* Linear Eval. Acc. = Freeze trained network and then attach one fully connected layer at the end of the network.  

## Issues
- SOP dataset does not converges, I'm trying to figure it out.  

## TODO LIST

- [x] *Known as `Proxy-NCA`*, No Fuss Distance Metric Learning using Proxies, Y. Movshovitz-Attias et al., ICCV 2017
- [ ] Correcting the Triplet Selection Bias for Triplet Loss, B. Yu et al., ECCV 2018
- [ ] SoftTriple Loss: Deep Metric Learning Without Triplet Sampling, Q. Qian et al., ICCV 2019
- [ ] Proxy Anchor Loss for Deep Metric Learning, S. Kim et al., CVPR 2020
- [ ] ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis, E. W. Teh et. al., ECCV 2020


## References
1. [pytorch impl: https://github.com/dichotomies/proxy-nca](https://github.com/dichotomies/proxy-nca)