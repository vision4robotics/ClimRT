# Continuity-Aware Latent Interframe Information Mining for Reliable UAV Tracking

### Changhong Fu, Mutian Cai, Sihang Li, Kunhan Lu, Haobo Zuo and Chongjun Liu


## Abstract
Object tracking is crucial for the autonomous navigation of unmanned aerial vehicles (UAVs) and has broad application in robotic automation fields. However, reliable aerial tracking remains a challenging task due to various difficulties like frequent occlusion and aspect ratio change. Additionally, most of the existing work mainly focus on explicit information to improve tracking performance, ignoring potential connections between frames. To solve the above issues, this work proposes a novel framework with continuity-aware latent interframe information mining for reliable UAV tracking, i.e., ClimRT. Specifically, a new efficient continuity-aware latent interframe information mining network (ClimNet) is proposed for UAV tracking, which can generate highly-effective latent frames between two adjacent frames. Besides, a novel location-continuity Transformer (LCT) is designed to fully explore continuity-aware spatial-temporal information, thereby markedly enhancing UAV tracking. Extensive qualitative and quantitative experiments on three authoritative aerial benchmarks have strongly validated the robustness and reliability of ClimRT in UAV Tracking. Furthermore, real-world tests on the typical aerial platform have proven its practicability and effectiveness.

![Workflow of our tracker](https://github.com/cvmutian/ClimRT/blob/main/imgs/img1.png)

This figure shows the workflow of our tracker.

## About Code
### 1. Environment setup
The ClimNet has been tested on Ubuntu 18.04, Python 3.7.4, numpy==1.19.2 , Pytorch 1.5.0, torchvision==0.6.0, cudatoolkit==10.1.

The ClimRT has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirement.txt
```

### 2. Test
Download pretrained model for ClimNet: [model_best](https://pan.baidu.com/s/1QeU7OcTqHksZXscBq3skiw)(code: c99t) and put it into `./pretrained_models` directory.
Download pretrained model: [t_best](https://pan.baidu.com/s/1QeU7OcTqHksZXscBq3skiw)(code: c99t) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `Dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

```bash 
python test.py                                
	--dataset UAV123                #dataset_name
	--snapshot snapshot/t_best.pth  # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

### 3. Train

#### Prepare training datasets

Download the datasets：
* [Vimeo-90K](http://toflow.csail.mit.edu/)
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)


**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.


#### Train a model
To train the ClimNet model, run `main.py` with the desired configs:
```bash
python main.py --batch_size 32 --test_batch_size 32 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 100 --lr 0.002 --data_root /train_dataset/vimeo_triplet --n_outputs 1 

```
To train the ClimRT model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

### 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1RVSiq7XUJCQnyXtoRq9SYg) (code: tj12) of UAV123@10fps, UAV123 and UAVTrack112. If you want to evaluate the tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV123                  \ # dataset_name
	--tracker_prefix 't_best'   # tracker_name
```

### 5. Contact
If you have any questions, please contact me.

Mutian Cai

Email: [1951223@tongji.edu.cn](1951223@tongji.edu.cn)



## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot), [HiFT](https://github.com/vision4robotics/HiFT) and [FLAVR](https://github.com/tarun005/FLAVR). We would like to express our sincere thanks to the contributors.
