# 基于多模态学习的三维人体行为预测技术研究

本仓库实现基于多模态学习的人体行为预测技术研究

## Dependencies

- Python >= 3.7
- PyTorch >= 1.11.0
- NVIDIA Apex (for Infogcn)
- tqdm, tensorboardX, wandb

## Data Preparation

- 请遵照原始[Infogcn](https://github.com/stnoah1/infogcn)仓库中步骤完成骨架数据集准备
- 对于行为预测，我们使用`seg_1`后缀中的代码进行数据集切分。具体来说，超参数`prediction_ratio`控制切分比例，在`get_raw_denoised_data.py`中直接对去噪后的数据按比例切分保存

## Training & Testing

### Training
- 我们将Numpy和PyTorch的种子数设置为1，以确保结果的可重复性


```
python main.py --half=True --batch_size=96 --test_batch_size=96 --step 50 60 --num_epoch=70 --n_heads=3 --num_worker=4 --k=8 --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 --use_vel=False --datacase=NTU60_CS --weight_decay=0.0005 --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder
```

### Testing

- 要测试保存在 <work_dir> 中的训练模型，请运行以下命令：

```
python main.py --half=True --test_batch_size=96 --n_heads=3 --num_worker=4 --k=1 --dataset=ntu --num_class=60 --use_vel=False --datacase=NTU60_CS --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder --phase=test --save_score=True --weights=<path_to_weight>
```

## Acknowledgements

This repo is based on [InfoGCN](https://github.com/stnoah1/infogcn), [GAP](https://github.com/MartinXM/GAP) and [SA-DVAE](https://github.com/pha123661/SA-DVAE). The data processing is borrowed from [InfoGCN]. Thanks to the original authors for their work!

