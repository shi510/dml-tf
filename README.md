# dml-tf
distance metric learning, tf2 implementation.  

## Python Packages
1. tensorflow-v2.4.0  
2. tensorflow-addons  
3. numpy  

## How To Use
Before you train, configure train/config.py.  
```
git clone https://github.com/shi510/dml-tf
cd dml-tf
export PYTHONPATH=$PYTHONPATH:$(pwd)
python train/main.py
```

## What's Different from existing github implementations?
1. ProxyNCA Loss  
![ProxyNCA Loss](docs/images/proxynca_loss.jpg)  
ProxyNCA loss can be calculated by tf.nn.log_softmax.  
But we can not use the function directly because we should seperate positive and negative proxy distances.  
Also we should consider numerical stability of our log_softmax implementation.  
So, We can rewrite `log_softmax=log(exp(a)/sum(exp(b)))=a-log(sum(exp(b)))`.  
See [train/loss/proxynca.py](train/loss/proxynca.py) and [train/loss/utils.py:pairwise_distance](train/loss/utils.py).  

2. ProxyAnchor Loss  
![ProxyAnchor Loss](docs/images/proxyanchor_loss.jpg)  
ProxyAnchor loss can be calculated by tf.math.reduce_logsumexp.  
But we can not use the function directly as the same issues on proxyNCA Loss.  
So, this project re-implements the reduce_logsumexp seperating positive and negative proxy distances.  
Also, the function can be rewriten for numerical stability as below:  
`logsumexp(x) = c + log(sum(x-c)), where c is a maximum of x.`  
See [train/loss/proxyanchor.py](train/loss/proxyanchor.py).  

## Training Conditions
1. Augmentation: random_flip, random_color, random_cutout  
2. Random Crop: All images are resized by 256x256 and then random crop by 224x224.  
3. Scale Factors: `this is very important. (Depending on your dataset size)`  
4. All network should be pretrained by imagenet, `this is important.`  

**SOP dataset does not converges, I'm trying to figure it out.**  

## Results of `ProxyNCA` on Test Set
|                 | CUB         | cars196     | InShop      |
|-----------------|-------------|-------------|-------------|
| NMI             | 73%         | 69%         | 86%         |
| Recall @ 1      | 65%         | 76%         | 71%         |
| Recall @ 2      | 73%         | 83%         | 79%         |
| Recall @ 4      | 79%         | 87%         | 85%         |
| Recall @ 8      | 85%         | 91%         | 89%         |
| Epoch           | 13          | 16          | 14          |
| Arch.           | InceptionV3 | InceptionV3 | InceptionV3 |
| Batch Size      | 64          | 64          | 64          |
| Embedding Size  | 64          | 64          | 64          |
| Proxy Scale     | 8           | 8           | 8           |
| Embedding Scale | 1           | 1           | 1           |
| Weight Opt      | AdamW@1e-4  | AdamW@1e-4  | AdamW@1e-4  |
| Proxy Opt       | AdamW@1e+0  | AdamW@1e+1  | AdamW@1e+1  |

## TODO LIST
- [x] *Known as `Proxy-NCA`*, No Fuss Distance Metric Learning using Proxies, Y. Movshovitz-Attias et al., ICCV 2017
- [ ] Correcting the Triplet Selection Bias for Triplet Loss, B. Yu et al., ECCV 2018
- [ ] SoftTriple Loss: Deep Metric Learning Without Triplet Sampling, Q. Qian et al., ICCV 2019
- [ ] Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning, X. Wang et at., CVPR 2019
- [ ] Proxy Anchor Loss for Deep Metric Learning, S. Kim et al., CVPR 2020
- [ ] ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis, E. W. Teh et al., ECCV 2020

## References
1. [pytorch impl: https://github.com/dichotomies/proxy-nca](https://github.com/dichotomies/proxy-nca)