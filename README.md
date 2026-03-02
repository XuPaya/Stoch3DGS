
<p align="center">

  <h1 align="center">Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://xupaya.github.io/">Peiyu Xu</a>,
    <a href="https://www.sunxin.name/">Xin Sun</a>,
    <a href="https://krishnamullia.com/">Krishna Mullia</a>,
    <a href="https://raymondyfei.github.io/">Raymond Fei</a>,
    <a href="https://iliyan.com/">Iliyan Georgiev</a>,
    <a href="https://shuangz.com/">Shuang Zhao</a>,
  </p>
  <h2 align="center">CVPR2026</h2>

  <p align="center">
  <br>
    <a href="https://xupaya.github.io/stochGRT/"><strong>Project Page</strong></a>
    |
    <a href=""><strong>Paper (Coming Soon)</strong></a>
  </p>
</p>

This repository provides an implementations of **Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting (CVPR2026)**. We build on top of the official implementation of [3DGRUT](https://github.com/nv-tlabs/3dgrut) and provide an accelerated algorithm. We also provide scripts to reproduce results reported in our paper on the MipNeRF360 dataset.

## 1. Dependencies, Installation and Usage

Please refer to the original 3DGRUT repository for installation instructions and usage details. We provide additional scripts to train and evaluate our method on the MipNeRF360 dataset.

## 2. Baselines and Methods

We provide the following algorithms in our codebase:

- **3DGRT**: The original 3D Gaussian Ray Tracing algorithm proposed by Moenne-Loccoz et al ([https://arxiv.org/abs/2407.07090](https://arxiv.org/abs/2407.07090)), which uses a deterministic sorted ray tracing algorithm.

- **Stochastic GRT**: The stochastic ray tracing algorithm proposed in our paper, which uses a stochastic sampling strategy for both the **forward** rendering and the backward gradient computation.

- **Quasi-Stochastic GRT**: A variant of the stochastic ray tracing algorithm that uses **sorted** ray tracing for the **forward** pass and the **stochastic** strategy for the **backward** pass. Note that this algorithm was not included in the paper, but we provide it here for reference of optimal performance and future research.

## 3. Evaluations on MipNeRF360

We provide scripts to reproduce results reported in our publication. All results are collected using a single **RTX 5880 Ada Generation** GPU.

***3DGRT (Baseline) Results***

```bash
bash ./benchmark/mipnerf360.sh apps/colmap_3dgrt.yaml
bash ./benchmark/mipnerf360_render.sh results/mipnerf360
```
|           | PSNR  | SSIM	| Train (s) |
|-----------|-------|-------|-------|
| Bicycle   | 24.81	| 0.744	| 2607	|
| Bonsai    | 31.78  | 0.937 | 4971	|
| Counter   | 28.33  | 0.900	| 4643	|
| Garden    | 26.77  | 0.847 | 2955	|
| Kitchen   | 30.39  | 0.920	| 7679	|
| Room      | 30.78  | 0.914	| 3708	|
| Stump     | 26.40	| 0.771	| 2272	|
| *Average* | 28.47	| 0.862	| 4119	|

***Stochastic GRT Results***

```bash
bash ./benchmark/mipnerf360_stoch.sh apps/colmap_3dgrt_stoch.yaml
bash ./benchmark/mipnerf360_render.sh results/mipnerf360_stoch/
```
|           | PSNR  | SSIM	| Train (s) |
|-----------|-------|-------|-------|
| Bicycle   | 24.58	| 0.724	| 2005	|
| Bonsai    | 31.34	| 0.931	| 3379	|
| Counter   | 28.41	| 0.894	| 3445	|
| Garden    | 26.22	| 0.826	| 2135	|
| Kitchen   | 29.76	| 0.903	| 4164	|
| Room      | 30.41	| 0.908	| 2293	|
| Stump     | 26.29	| 0.767	| 1830	|
| *Average* | 28.14	| 0.850	| 2750	|

***Quasi-Stochastic GRT Results***

```bash
bash ./benchmark/mipnerf360_quasistoch.sh apps/colmap_3dgrt_quasistoch.yaml
bash ./benchmark/mipnerf360_render.sh results/mipnerf360_quasistoch/
```
|           | PSNR  | SSIM	| Train (s) |
|-----------|-------|-------|-------|
| Bicycle   | 24.63	| 0.727	| 1570	|
| Bonsai    | 31.59	| 0.936	| 2407	|
| Counter   | 28.35	| 0.899	| 2223	|
| Garden    | 26.74 | 0.840	| 1796	|
| Kitchen   | 30.00	| 0.911	| 2883	|
| Room      | 30.39	| 0.909	| 1765	|
| Stump     | 26.50	| 0.774	| 1434	|
| *Average* | 28.31	| 0.857	| 2011	|
