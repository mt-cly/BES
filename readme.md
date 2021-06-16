# BES
PSA[1](#cankao)
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
* There are some slight differences in the implementation. In the 'synthesize boundary labels' (make_boundary_label.py), the filter principle will additionally consider the boundary between foregrounds. The formula is revised as follows:

<div align="center"><img src="https://latex.codecogs.com/png.latex?\hat{B}_{i}=\left\{&space;\begin{matrix}&space;{0}&space;&&space;\text{if}&space;\&space;\min&space;\{&space;\max&space;\limits_{c\in&space;\mathcal{C}}&space;S_{i}^{c}&space;,&space;S_{i}^{0}&space;\}&space;&&space;>&space;&&space;2\&space;\theta_{scale}&space;\\&space;{}&space;&&space;\text{&space;and}&space;\&space;|\max&space;\limits_{c&space;\in&space;\mathcal{C}}&space;S_{i}^{c}&space;-&space;S_{i}^{0}|&space;&&space;\geq&space;&&space;2\&space;\theta_{diff}&space;\\&space;{1}&space;&&space;\text{if&space;}&space;\&space;\min&space;\{&space;\max&space;\limits_{c\in&space;\mathcal{C}}&space;S_{i}^{c}&space;,&space;S_{i}^{0}&space;\}&space;&&space;>&space;&&space;\theta_{scale}&space;\\&space;{}&space;&&space;\text{&space;and}&space;\&space;|\max&space;\limits_{c&space;\in&space;\mathcal{C}}S&space;_{i}^{c}-&space;S_{i}^{0}|&space;&&space;<&space;&&space;\theta_{diff}&space;\\&space;{1}&space;&&space;\text{if&space;}&space;\&space;\max&space;\limits_{c&space;\in&space;\mathcal{C}-&space;\arg&space;\max&space;S_i^c}&space;S&space;_{i}^{c}&space;&&space;>&space;&&space;\theta_{scale}&space;\\&space;{}&space;&&space;\text{&space;and}&space;\&space;\max&space;\limits_{c&space;\in&space;\mathcal{C}}S&space;_{i}^{c}-&space;\max&space;\limits_{c&space;\in&space;\mathcal{C}-&space;\arg&space;\max&space;S_i^c}&space;S&space;_{i}^{c}&space;&&space;<&space;&&space;\theta_{diff}&space;\\&space;{255}&space;&&space;\text&space;{otherwise}&space;&&space;\end{matrix}&space;\right."/></div>

* The generated pseudo semantic segmentation labels will used to be provide supervision for DeepLab_v1 and DeepLab_v2. There are many considerable implementation in the github, for example: [DeepLab-v1](https://github.com/wangleihitcs/DeepLab-V1-PyTorch), [DeepLab-v2](https://github.com/kazuto1011/deeplab-pytorch), and so on.

* In previous experiments, the original seed is not set and may cause some bias between reproduction and paper report. Here I report the BES performance in the PASCAL VOC 2012 training set with different seeds.

<div align="center">
<table>
    <thead><tr>
        <th>seed</th> <th>report in paper</th>
        <th>0</th> <th>1</th> <th>2</th> <th>3</th> <th>4</th> <th>5</th> <th>6</th> <th>7</th> <th>8</th> <th>9</th>
      </tr>
    </thead>
      <tbody><tr>
        <td> mIoU (w\o dCRF)</td> <td>66.4</td>
        <td>66.2</td> <td>66.5</td> <td>66.2</td> <td>66.4</td> <td>66.4</td> <td>67.9</td> <td>66.0</td> <td>66.7</td> <td>65.7</td> <td>66.6</td>
      </tr></tbody>
  </table>
</div>

* This paper is inspired by the PSA[1](#cankao) and IRN[^2]. In my view, the boundary is implicitly explored  through the prediction of  pixels-affinity, BES can be regarded as another implementation.

cankao:Ahn J, Kwak S. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4981-4990.
[^2]:Ahn J, Cho S, Kwak S. Weakly supervised learning of instance segmentation with inter-pixel relations[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 2209-2218.

