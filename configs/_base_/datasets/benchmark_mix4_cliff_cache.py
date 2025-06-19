"""
dataset settings from 
https://github.com/smplbody/hmr-benchmarks/blob/dev/configs/combine/hrnet_hmr_mix4_coco_l1_aug.py

loss_keypoints3d=dict(type='L1Loss', loss_weight=100),
loss_keypoints2d=dict(type='L1Loss', loss_weight=10),
loss_vertex=dict(type='L1Loss', loss_weight=2),
loss_smpl_pose=dict(type='L1Loss', loss_weight=3),
loss_smpl_betas=dict(type='L1Loss', loss_weight=0.02),
loss_adv=dict(
    type='GANLoss',
    gan_type='lsgan',
    real_label_val=1.0,
    fake_label_val=0.0,
    loss_weight=1),
disc=dict(type='SMPLDiscriminator'))
"""

# dataset settings
img_resolution = 256
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'has_keypoints3d', 'has_keypoints2d', 'smpl_body_pose',
    'smpl_global_orient', 'smpl_betas', 'smpl_transl', 'keypoints2d',
    'keypoints3d', 'sample_idx', 'img_h', 'img_w', 'focal_length', 'center',
    'scale', 'bbox_info', 'crop_trans', 'inv_trans'
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.2),
    dict(type='SyntheticOcclusion', occluders_file='data/occluders/pascal_occluders.npy'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_54'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_resolution),
    dict(type='MixingErasing', ntype='self', mlist=['h36m']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'bbox_xywh',
            'ori_shape'])
            ]

adv_data_keys = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_adv_pipeline = [dict(type='Collect', keys=adv_data_keys, meta_keys=[])]

test_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',keys=['img', *data_keys],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'bbox_xywh',
            'ori_shape'
        ])]

cache_files = dict(
    h36m='data/cache/h36m_mosh_train_smpl_54.npz',
    coco='data/cache/eft_coco_train_smpl_54.npz',
    lspet='data/cache/eft_lspet_train_smpl_54.npz',
    mpii='data/cache/eft_mpii_train_smpl_54.npz',
    mpi_inf_3dhp='data/cache/spin_mpi_inf_3dhp_train_smpl_54.npz')

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=12,
    train=dict(
        type='AdversarialDataset',
        train_dataset=dict(
            type='MixedDataset',
            configs=[
                dict(
                    type=dataset_type,
                    dataset_name='h36m',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['h36m'],
                    convention='smpl_54',
                    ann_file='h36m_mosh_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='coco',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['coco'],
                    convention='smpl_54',
                    ann_file='eft_coco_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='lspet',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['lspet'],
                    convention='smpl_54',
                    ann_file='eft_lspet_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='mpii',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['mpii'],
                    convention='smpl_54',
                    ann_file='eft_mpii_train.npz'),
                dict(
                    type=dataset_type,
                    dataset_name='mpi_inf_3dhp',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['mpi_inf_3dhp'],
                    convention='smpl_54',
                    ann_file='spin_mpi_inf_3dhp_train.npz'),
            ],
            partition=[0.5, 0.233, 0.046, 0.021, 0.2],
        ),
        adv_dataset=dict(
            type='MeshDataset',
            dataset_name='cmu_mosh',
            data_prefix='data',
            pipeline=train_adv_pipeline,
            ann_file='cmu_mosh.npz')),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
)
