# Activity-prediction-based-on-multimodal-learning

This repository focus on the research on human action prediction technology based on multimodal learning.

## Dependencies

- Python >= 3.7
- PyTorch >= 1.11.0
- NVIDIA Apex (for Infogcn)
- tqdm, tensorboardX, wandb

## Data Preparation

- Please follow the steps in the original [Infogcn] (https://github.com/stnoah1/infogcn) repository to complete the preparation of the skeleton dataset **NTU**.
- For action prediction, we use the code in the `seg_1` suffix for dataset segmentation. Specifically, the hyperparameter `prediction_ratio` controls the split ratio. The data is split and saved proportionally in `get_raw_denoised_data.py` after denoising.

## Training & Testing

### Training
- We set the seed number for Numpy and PyTorch as 1 for reproducibility.


```
python main.py --half=True --batch_size=96 --test_batch_size=96 --step 50 60 --num_epoch=70 --n_heads=3 --num_worker=4 --k=8 --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 --use_vel=False --datacase=NTU60_CS --weight_decay=0.0005 --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --half=True --test_batch_size=96 --n_heads=3 --num_worker=4 --k=1 --dataset=ntu --num_class=60 --use_vel=False --datacase=NTU60_CS --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder --phase=test --save_score=True --weights=<path_to_weight>
```

## Acknowledgements

This repo is based on [InfoGCN](https://github.com/stnoah1/infogcn), [GAP](https://github.com/MartinXM/GAP) and [SA-DVAE](https://github.com/pha123661/SA-DVAE). The data processing is borrowed from [InfoGCN]. Thanks to the original authors for their work!

