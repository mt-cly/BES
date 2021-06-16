# BES
> paper: Weakly Supervised Semantic Segmentation with Boundary Exploration. [[pdf]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710341.pdf)

In this paper, we propose a two-stage framework to tackle semantic segmentation problem with image-level annotation supervision. In the first stage, Attention-pooling CAM is adopted to obtain coarse localization cues, which will be used to synthesize pseudo object boundary labels, in second stage, the prediction of object boundaries will be refined through BENet and direct the propagation of CAM results.

## USAGE
This code heavily depends on the [IRN](https://github.com/jiwoon-ahn/irn). 
#### Preparation
* Dataset: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) & [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
* Python: 3.6
#### Install python packages
```python
 pip install -r requirement.txt
```
For pytorch, installation command can be easily found with specified setting on [official website](https://pytorch.org/get-started/locally/).

#### Run the code
Specify the VOC dataset path and run the command
```python
python run_sample.py --voc12_root xxxxx
```

## Tips
* There are some slight differences in the implementation. In the 'synthesize boundary labels' step (make_boundary_label.py), the filter principle will additionally consider the boundary between foregrounds. The formula is revised as follows:

<div align="center"><img src="http://www.sciweavers.org/tex2img.php?eq=%5Chat%7BB%7D_%7Bi%7D%3D%5Cleft%5C%7B%0A%5Cbegin%7Bmatrix%7D%0A%7B0%7D%20%26%20%5Ctext%7Bif%7D%20%5C%20%5Cmin%20%5C%7B%20%5Cmax%20%5Climits_%7Bc%5Cin%20%5Cmathcal%7BC%7D%7D%20S_%7Bi%7D%5E%7Bc%7D%20%2C%20%20S_%7Bi%7D%5E%7B0%7D%20%5C%7D%20%20%26%20%3E%20%26%202%5C%20%5Ctheta_%7Bscale%7D%20%5C%5C%0A%7B%7D%20%26%20%5Ctext%7B%20and%7D%20%5C%20%7C%5Cmax%20%5Climits_%7Bc%20%5Cin%20%5Cmathcal%7BC%7D%7D%20S_%7Bi%7D%5E%7Bc%7D%20-%20S_%7Bi%7D%5E%7B0%7D%7C%20%26%20%5Cgeq%20%26%202%5C%20%5Ctheta_%7Bdiff%7D%20%5C%5C%0A%7B1%7D%20%26%20%5Ctext%7Bif%20%7D%20%5C%20%5Cmin%20%5C%7B%20%5Cmax%20%5Climits_%7Bc%5Cin%20%5Cmathcal%7BC%7D%7D%20S_%7Bi%7D%5E%7Bc%7D%20%2C%20%20S_%7Bi%7D%5E%7B0%7D%20%5C%7D%20%26%20%3E%20%26%20%5Ctheta_%7Bscale%7D%20%5C%5C%0A%7B%7D%20%26%20%5Ctext%7B%20and%7D%20%5C%20%7C%5Cmax%20%5Climits_%7Bc%20%5Cin%20%5Cmathcal%7BC%7D%7DS%20_%7Bi%7D%5E%7Bc%7D-%20S_%7Bi%7D%5E%7B0%7D%7C%20%26%20%3C%20%26%20%5Ctheta_%7Bdiff%7D%20%5C%5C%0A%7B1%7D%20%26%20%5Ctext%7Bif%20%7D%20%5C%20%5Cmax%20%5Climits_%7Bc%20%5Cin%20%5Cmathcal%7BC%7D-%20%5Carg%20%5Cmax%20S_i%5Ec%7D%20S%20_%7Bi%7D%5E%7Bc%7D%20%26%20%3E%20%26%20%5Ctheta_%7Bscale%7D%20%5C%5C%0A%7B%7D%20%26%20%5Ctext%7B%20and%7D%20%5C%20%5Cmax%20%5Climits_%7Bc%20%5Cin%20%5Cmathcal%7BC%7D%7DS%20_%7Bi%7D%5E%7Bc%7D-%20%5Cmax%20%5Climits_%7Bc%20%5Cin%20%5Cmathcal%7BC%7D%20-%20%5Carg%20%5Cmax%20S_i%5Ec%7D%20S%20_%7Bi%7D%5E%7Bc%7D%20%26%20%3C%20%26%20%5Ctheta_%7Bdiff%7D%20%5C%5C%0A%7B255%7D%20%26%20%5Ctext%20%7Botherwise%7D%20%26%0A%5Cend%7Bmatrix%7D%0A%5Cright.&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\hat{B}_{i}=\left\{\begin{matrix}{0} & \text{if} \ \min \{ \max \limits_{c\in \mathcal{C}} S_{i}^{c} ,  S_{i}^{0} \}  & > & 2\ \theta_{scale} \\{} & \text{ and} \ |\max \limits_{c \in \mathcal{C}} S_{i}^{c} - S_{i}^{0}| & \geq & 2\ \theta_{diff} \\{1} & \text{if } \ \min \{ \max \limits_{c\in \mathcal{C}} S_{i}^{c} ,  S_{i}^{0} \} & > & \theta_{scale} \\{} & \text{ and} \ |\max \limits_{c \in \mathcal{C}}S _{i}^{c}- S_{i}^{0}| & < & \theta_{diff} \\{1} & \text{if } \ \max \limits_{c \in \mathcal{C}- \arg \max S_i^c} S _{i}^{c} & > & \theta_{scale} \\{} & \text{ and} \ \max \limits_{c \in \mathcal{C}}S _{i}^{c}- \max \limits_{c \in \mathcal{C} - \arg \max S_i^c} S _{i}^{c} & < & \theta_{diff} \\{255} & \text {otherwise} &\end{matrix}\right." width="451" height="210" /></div>

* The generated pseudo semantic segmentation labels will used to be provide supervision for DeepLab_v1 and DeepLab_v2. There are many considerable implementation in the github, for example: [DeepLab-v1](https://github.com/wangleihitcs/DeepLab-V1-PyTorch), [DeepLab-v2](https://github.com/kazuto1011/deeplab-pytorch), and so on.

* In previous experiments, the original seed is not set and may cause some bias between reproduction and paper report. Here I report the BES performance in the PASCAL VOC 2012 training set with different seeds.
|seed|report in paper|0|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|---|---|---|
|mIoU (w\o dCRF)|66.4|66.2|66.5|66.2|66.4|66.4|67.9|66.0|66.7|65.7|66.6|

* This paper is inspired by the PSA[^1] and IRN[^2]. In my view, the boundary is implicitly explored  through the prediction of  pixels-affinity, BES can be regarded as another implementation.

[^1]:Ahn J, Kwak S. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4981-4990.
[^2]:Ahn J, Cho S, Kwak S. Weakly supervised learning of instance segmentation with inter-pixel relations[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 2209-2218.

