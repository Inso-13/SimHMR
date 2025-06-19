"""
finetune_config
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
file_client_args=dict(backend='disk')

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
                    dataset_name='pw3d',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path="data/cache/pw3d_train_smpl_49.npz",
                    convention='smpl_49',
                    ann_file='data/preprocessed_datasets/pw3d_train.npz'),
            ],
            partition=[1.0],
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
