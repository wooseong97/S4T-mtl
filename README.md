# Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training [ICCV 2025]

**Official PyTorch implementation of [*Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training*](https://arxiv.org/abs/2507.07778) [ICCV 2025].**

**Wooseong Jeong\***, **Jegyeong Cho\***, **Youngho Yoon\***, and **Kuk-Jin Yoon**
*Visual Intelligence Lab., KAIST, Korea*
📧 stk14570@kaist.ac.kr, j2k0618@kaist.ac.kr, dudgh1732@kaist.ac.kr, kjyoon@kaist.ac.kr

Generalizing neural networks to unseen target domains is a significant challenge in real-world deployments. Test-time training (TTT) addresses this by using an auxiliary self-supervised task to reduce the domain gap caused by distribution shifts between the source and target. However, we find that when models are required to perform multiple tasks under domain shifts, conventional TTT methods suffer from unsynchronized task behavior, where the adaptation steps needed for optimal performance in one task may not align with the requirements of other tasks. To address this, we propose a novel TTT approach called Synchronizing Tasks for Test-time Training (S4T), which enables the concurrent handling of multiple tasks. The core idea behind S4T is that predicting task relations across domain shifts is key to synchronizing tasks during test time. To validate our approach, we apply S4T to conventional multi-task benchmarks, integrating it with traditional TTT protocols. Our empirical results show that S4T outperforms state-of-the-art TTT methods across various benchmarks.

## Method overview

S4T is a two-stage pipeline built on a multi-task network:

1. **Source-domain pre-training.** A shared backbone produces per-task predictions. The backbone features are additionally compressed into per-task feature maps that feed (i) auxiliary task heads and (ii) a joint MAE-style decoder that reconstructs all task outputs from a masked input. The reconstruction branch provides a self-supervised objective.
2. **Test-time training (TTT).** Starting from the pre-trained checkpoint, the model is adapted on the target domain using only the self-supervised consistency / reconstruction objective (no target labels).

## Installation

```bash
conda env create -f environment.yaml
conda activate s4t
```

## Datasets

We use the standard multi-task dense-prediction benchmarks **PASCAL-Context** and **NYUDv2**, together with the **Taskonomy (tiny)** split. Set the dataset roots in [`configs/mypath.py`](configs/mypath.py):

```python
db_root           = "../data/"        # contains PASCALContext/ and NYUDv2/
db_root_taskonomy = "../datasets/"    # contains taskonomy/data/
```

## Usage

Configurations live under [`configs/S4T/`](configs/S4T): `pretrain/` for source-domain pre-training and `ttt/` for test-time training on the target domain. Each TTT config points to the pre-trained checkpoint via its `pretrained_ckpt:` field — update the `out_dir`/`pretrained_ckpt` paths to your environment.

**1. Source-domain pre-training**

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29500 \
    main.py --config_exp ./configs/S4T/pretrain/taskonomy_nyud_resnet50_S4T.yml --run_mode train --seed 1
```

**2. Test-time training on the target domain**

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29500 \
    main.py --config_exp ./configs/S4T/ttt/taskonomy_nyud_resnet50_S4T.yml --run_mode train --seed 1
```

See [`run.sh`](run.sh) for a runnable example.

### Test-time training protocol

At test time the model loads the pre-trained checkpoint, draws a batch of **unlabeled** target data, resets to the checkpoint (`episodic`), and adapts for a **fixed** number of self-supervised steps (`steps` in the config). It is then evaluated **once** on the target test set. No target labels are used during adaptation, and the number of adaptation steps is fixed in advance (it is not selected on the test set). Set `--seed` for reproducibility.

## Acknowledgements

This codebase builds on prior multi-task dense-prediction code, including [MTI-Net](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch) (Simon Vandenhende), released under the CC BY-NC 4.0 license. We thank the original authors for making their code available.

## License

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license, inherited from the base codebases. See [`LICENSE`](LICENSE).

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{jeong2025s4t,
  title     = {Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training},
  author    = {Jeong, Wooseong and Cho, Jegyeong and Yoon, Youngho and Yoon, Kuk-Jin},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

## Contact

Wooseong Jeong: stk14570@kaist.ac.kr
