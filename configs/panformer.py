# ---> GENERAL CONFIG <---
name = 'panformer_GF-2'
description = 'test panformer on PSData3/GF-2 dataset'

model_type = 'PanFormer'
work_dir = f'data/PSData3/model_out/{name}'
log_dir = f'logs/{model_type.lower()}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = True
checkpoint = f'data/PSData3/model_out/{name}/train_out/pretrained.pth'

# ---> DATASET CONFIG <---
aug_dict = {'lr_flip': 0.5, 'ud_flip': 0.5}
ms_chans = 4
bit_depth = 10
train_set_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/PSData3/Dataset/GF-2/train_low_res'],
        bit_depth=10),
    num_workers=4,
    batch_size=4,
    shuffle=True)
test_set0_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/PSData3/Dataset/GF-2/test_full_res'],
        bit_depth=10),
    num_workers=4,
    batch_size=1,
    shuffle=False)
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/PSData3/Dataset/GF-2/test_low_res'],
        bit_depth=10),
    num_workers=4,
    batch_size=1,
    shuffle=False)
seed = 19971118
cuda = True
max_iter = 200000
save_freq = 10000
test_freq = 10000
eval_freq = 10000
norm_input = False

# ---> SPECIFIC CONFIG <---
optim_cfg = {
    'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=1e-4),
}
sched_cfg = dict(step_size=10000, gamma=0.99)
loss_cfg = {
    'rec_loss': dict(type='l1', w=1.)
}
model_cfg = {
    'core_module': dict(n_feats=64, n_heads=8, head_dim=8, win_size=4, n_blocks=3,
                        cross_module=['pan', 'ms'], cat_feat=['pan', 'ms']),
}
