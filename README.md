# Stable Mean Teacher for Semi-supervised Video Action Detection


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stable-mean-teacher-for-semi-supervised-video/semi-supervised-video-action-detection-on)](https://paperswithcode.com/sota/semi-supervised-video-action-detection-on?p=stable-mean-teacher-for-semi-supervised-video) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stable-mean-teacher-for-semi-supervised-video/action-detection-on-ucf101-24)](https://paperswithcode.com/sota/action-detection-on-ucf101-24?p=stable-mean-teacher-for-semi-supervised-video)

[Akash Kumar](https://akash2907.github.io/), Sirshapan Mitra, [Yogesh S Rawat](https://www.crcv.ucf.edu/person/rawat/)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.07072)


Official code for our paper "Stable Mean Teacher for Semi-supervised Video Action Detection"

## :rocket: News
* **(Dec 12, 2024)**
  * This paper has been accepted for publication in the [AAAI 2025 Main Technical Track](https://aaai.org/conference/aaai/aaai-25/main-technical-track/) conference.
* **(Jan 05, 2025)**
  * Project website is now live at [smt-webpage](https://akash2907.github.io/smt_webpage/)
* **(Feb 25, 202)**
  * Code for our method on semi-supervised video action detection has been released.

<hr>


![method-diagram](https://akash2907.github.io/smt_webpage/static/images/smt.png)
> **Abstract:** *In this work, we focus on semi-supervised learning for video action detection. Video action detection requires spatiotemporal localization in addition to classification, and a limited amount of labels makes the model prone to unreliable predictions. We present Stable Mean Teacher, a simple end-to-end teacher-based framework that benefits from improved and temporally consistent pseudo labels. It relies on a novel Error Recovery (EoR) module, which learns from students' mistakes on labeled samples and transfers this knowledge to the teacher to improve pseudo labels for unlabeled samples. Moreover, existing spatiotemporal losses do not take temporal coherency into account and are prone to temporal inconsistencies. To address this, we present Difference of Pixels (DoP), a simple and novel constraint focused on temporal consistency, leading to coherent temporal detections. We evaluate our approach on four different spatiotemporal detection benchmarks: UCF101-24, JHMDB21, AVA, and YouTube-VOS. Our approach outperforms the supervised baselines for action detection by an average margin of 23.5% on UCF101-24, 16% on JHMDB21, and 3.3% on AVA. Using merely 10% and 20% of data, it provides competitive performance compared to the supervised baseline trained on 100% annotations on UCF101-24 and JHMDB21, respectively. We further evaluate its effectiveness on AVA for scaling to large-scale datasets and YouTube-VOS for video object segmentation, demonstrating its generalization capability to other tasks in the video domain. Code and models are publicly available.*
>

## :trophy: Achievements and Features

- We establish **state-of-the-art results (SOTA)** in semi-supervised video action detection on UCF101-24, JHMDB-21 and real-time approach on AVA.
- We propose a class-agnostic error localization and temporal constraints that helps provide stronger pseudo labels and rectify fine-grained spatio-temporal localization mistakes.

## :hammer_and_wrench: Setup and Installation
We have used `python=3.8.16`, and `torch=1.10.0` for all the code in this repository. It is recommended to follow the below steps and setup your conda environment in the same way to replicate the results mentioned in this paper and repository.

1. Clone this repository into your local machine as follows:
```bash
git clone https://github.com/AKASH2907/stable-mean-teacher.git
```
2. Change the current directory to the main project folder :
```bash
cd stable-mean-teacher
```
3. To install the project dependencies and libraries, use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and install the defined environment from the .yml file by running:
```bash
conda env create -f environment.yml
```
4. Activate the newly created conda environment:
```bash
conda activate ssla 
```
### Datasets
To download and setup the required datasets used in this work, please follow these steps:
1. Download the UCF101-24 dataset from their official [website](https://www.crcv.ucf.edu/data/UCF101.php). 
2. Download extra annotation files/splits from: [UCF101-24-splits](https://drive.google.com/drive/u/0/folders/1aFlPKtzWIufyAOkcAmUySH4PB_uCPDkj).
3. Download the JHMDB21 dataset from their official [website](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). 
4. Download extra annotation files/splits from: [JHMDB21-splits](https://drive.google.com/drive/u/0/folders/1whGR2pg299D5W7jDV9Rop_jpr1ENIALF).

<!-- 
- `coco/`
  - `annotations/`
    - `instances_train2017.json`
    - `instances_val2017.json`
    - `ovd_instances_train2017_base.json`
    - `ovd_instances_val2017_basetarget.json`
    - `..other coco annotation json files (optional)..`
  - `train2017/`
  - `val2017/`
  - `test2017/`
- `lvis/`
  - `lvis_v1_val.json`
  - `lvis_v1_train.json`
  - `lvis_v1_val_subset.json` -->

 ### Model Weights
 All the pre-trained model weights can be downloaded from this link: [model weights](https://huggingface.co/akashkumar29/stable-mean-teacher/tree/main). 

 - **I3D_weights.pth**: I3D model weights used for Video Action detection task for intitalization is found at this [link](https://github.com/piergiaj/pytorch-i3d/tree/master/models).
 
### Training
Run the following script from the main project directory as follows:
1. JHMDB-21
   ```bash
   python semi_loc_feat_const_pa_stn_aug_add_aux_dop_jhmdb.py --epochs 50 --bs 8 --lr 1e-4 --pkl_file_label train_annots_10_labeled_random.pkl --pkl_file_unlabel train_annots_90_unlabeled_random.pkl --wt_loc 1 --wt_cls 1 --wt_cons 0.1 --const_loss l2 --thresh_epoch 11 -at 2 -ema 0.99 --opt3 --opt4 --ramp_thresh 0 --scheduler --exp_id 10_per/semi_loc_const_l2_thresh_10_eps_per10_aug_st_aux_op_raw_ramp_const_thresh_0_l2_dop_temporal_ramp_up_both_dop_l2_till_0_w_scheduler 
   ```
2. UCF101-24
    ```bash
   python semi_loc_feat_const_pa_stn_aug_add_aux_dop_ucf.py --epochs 50 --bs 8 --lr 1e-4 --txt_file_label jhmdb_classes_list_per_20_labeled.txt --txt_file_unlabel jhmdb_classes_list_per_80_unlabeled.txt --wt_loc 1 --wt_cls 1 --wt_cons 0.1 --const_loss l2 --thresh_epoch 11 -at 2 -ema 0.99 --opt3 --opt4 --ramp_thresh 16 --exp_id semi_loc_const_l2_thresh_10_eps_per20_aug_st_aux_op_raw_ramp_const_thresh_15_l2_dop_temporal_ramp_up_both_dop_l2_till_15 
   ```

* `--epochs`: Number of epochs.
* `--bs`: batch size.
* `--lr`: Learning rate.
* `--pkl_file_label/txt_file_label`: Labeled set video list.
* `--pkl_file_unlabel/txt_file_unlabel`: Unlabeled set video list.
* `--wt_loc/wt_cls/wt_cons`: Localization/Classification/Consistency loss weights.
* `--thresh_epoch/ramp_thresh`: Threshold epoch to model reach it's confidence (ignore labeled set consistency loss till then)/Ramping up loss epoch.
* `--opt3/opt4`: Ramp up DoP + L2 both, based on ramp thresh epochs/Both main+aux loss added same weight w/ any rampup.
* `--at`: Aug type: 0-spatial, 1- temporal, 2 - both.

### Evaluation

Run the following script from the main project directory as follows:
1. JHMDB-21
   ```bash
   ython multi_model_evalCaps_jhmdb.py --ckpt EXP-FOLDER-NAME 
   ```
2. UCF101-24
    ```bash
   python multi_model_evalCaps_ucf.py --ckpt EXP-FOLDER-NAME 
   ```
* `--ckpt`: Checkpoint Path.

## :medal_military: Semi-Supervised Action Detection Results on UCF101-24 and JHMDB21

This table presents the performance of various semi-supervised action detection approaches on the **UCF101-24** and **JHMDB21** datasets using the I3D backbone. The table reports results for different annotation percentages, comparing f@0.5, v@0.2, and v@0.5 metrics.

### Results Table
| **Semi-Supervised Approaches** | **Backbone** | **Annot.** | **UCF101-24** f@0.5 | v@0.2 | v@0.5 | **JHMDB21** Annot. | f@0.5 | v@0.2 | v@0.5 |
|--------------------------------|-------------|------------|----------------|------|------|------------|------|------|------|
| MixMatch (Berthelot et al. 2019) | I3D | 10% | 10.3 | 54.7 | 4.9 | 30% | 7.5 | 46.2 | 5.8 |
| Pseudo-label (Lee et al. 2013) | I3D | 10% | 59.3 | 89.9 | 58.3 | 20% | 55.3 | 87.6 | 52.0 |
| ISD (Jeong et al. 2021) | I3D | 10% | 60.2 | 91.3 | 64.0 | 20% | 57.8 | 90.2 | 57.0 |
| E2E-SSL (Kumar and Rawat 2022) | I3D | 10% | 65.2 | 91.8 | 66.7 | 20% | 59.1 | 93.2 | 58.7 |
| Mean Teacher (Tarvainen and Valpola 2017) | I3D | 10% | 67.3 | 92.7 | 70.5 | 20% | 56.3 | 88.8 | 52.8 |
| **Stable Mean Teacher (Ours)** | I3D | 10% | **73.9** | **95.8** | **76.3** | 20% | **69.8** | **98.8** | **70.7** |
| | | | _(↑ 6.6)_ | _(↑ 3.1)_ | _(↑ 5.8)_ | | _(↑ 13.5)_ | _(↑ 10.0)_ | _(↑ 17.9)_ |

### Notes:
- `f@0.5`: Frame-level detection accuracy at IoU 0.5.
- `v@0.2`: Video-level detection accuracy at IoU 0.2.
- `v@0.5`: Video-level detection accuracy at IoU 0.5.

## :framed_picture: Qualitative Visualization
![qual-analysis-1](https://akash2907.github.io/smt_webpage/static/images/smt_qual_analysis_visual.png)

## :email: Contact
Should you have any questions, please create an issue in this repository or contact at akash.kumar@ucf.edu

## :pray: Acknowledgement
Our code is built on [E2E-SSL](https://github.com/AKASH2907/pi-consistency-activity-detection). If you find our work useful, consider checking out that work.  

## :black_nib: Citation
If you found our work helpful, please consider starring the repository ⭐⭐⭐ and citing our work as follows:
```bibtex
@article{kumar2024stable,
      title={Stable Mean Teacher for Semi-supervised Video Action Detection},
      author={Kumar, Akash and Mitra, Sirshapan and Rawat, Yogesh Singh},
      journal={arXiv preprint arXiv:2412.07072},
      year={2024}
    }
```

```bibtex
@InProceedings{Kumar_2022_CVPR,
      author    = {Kumar, Akash and Rawat, Yogesh Singh},
      title     = {End-to-End Semi-Supervised Learning for Video Action Detection},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2022},
      pages     = {14700-14710}
  }
```

