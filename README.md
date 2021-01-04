# dml-tf
distance metric learning, tf2 implementation


## Issue

1. Convergence on MNIST dataset is good.  
2. Convergence on CIFAR10 dataset is not good, NMI saturates at 52%.  
3. Convergence on CARS196 dataset is totally not good, NMI is not improving.  

## TODO LIST

- [x] *Known as `Proxy-NCA`*, No Fuss Distance Metric Learning using Proxies, Y. Movshovitz-Attias et al., ICCV 2017
- [ ] Correcting the Triplet Selection Bias for Triplet Loss, B. Yu et al., ECCV 2018
- [ ] SoftTriple Loss: Deep Metric Learning Without Triplet Sampling, Q. Qian et al., ICCV 2019
- [ ] Proxy Anchor Loss for Deep Metric Learning, S. Kim et al., CVPR 2020
- [ ] ProxyNCA++: Revisiting and Revitalizing Proxy Neighborhood Component Analysis, E. W. Teh et. al., ECCV 2020