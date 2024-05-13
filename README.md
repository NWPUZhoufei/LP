# MeCo-net
Code and model for MeCo-net

# Requirements
- numpy  1.21.2
- scipy  1.3.0
- torch  1.6.0
- torchvision  0.7.0
- python 3.7.3

# Train
```
CUDA_VISIBLE_DEVICES=0  nohup python train.py --crop_num 6 --lamba_diversity_loss 0.3 --seed 1111 --epoch 50 --save_freq 50 --train_n_eposide 100 --n_support 5 --source_data_path ./source_domain/miniImageNet/train  --pretrain_model_path ./pretrain/400.tar --save_dir checkpoint >record.log 2>&1 &

```


# Test
```
- 5-way 1-shot:

eg:

EuroSAT:
CUDA_VISIBLE_DEVICES=0  nohup python test.py --n_support 1 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600  --model_path ./checkpoint/50.tar  >record_t.log 2>&1 &


- 5-way 5-shot:

eg:

EuroSAT:
CUDA_VISIBLE_DEVICES=0  nohup python test.py --n_support 5 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600  --model_path ./checkpoint/50.tar  >record_t.log 2>&1 &


```

# Transductive test
```

- 5-way 1-shot:

eg:

EuroSAT:

CUDA_VISIBLE_DEVICES=0  nohup python test_tr.py --n_support 1 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600   --model_path ./checkpoint/50.tar  >record_t.log 2>&1 &

- 5-way 5-shot:

eg:

EuroSAT:

CUDA_VISIBLE_DEVICES=0  nohup python test_tr.py --n_support 5 --seed 1111 --current_data_path ./target_domain/EuroSAT  --current_class 10 --test_n_eposide 600   --model_path ./checkpoint/50.tar  >record_t.log 2>&1 &

```
