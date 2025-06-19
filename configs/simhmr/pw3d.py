# SimHMR configuration for 3DPW dataset
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
use_adversarial_train = True
evaluation = dict(interval=1, metric=['pa-mpjpe', 'mpjpe'])
img_res = 256
optimizer = dict(
    backbone=dict(type='Adam', lr=0.0002), head=dict(type='Adam', lr=0.0002))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10,15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
find_unused_parameters = False
model = dict(
    type='SimHMR',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        norm_eval=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='data/pretrained/backbone/resnet50_coco_pose.pth')),
    head=dict(
        type='SimHMRHead',
        transformer=dict(
            type='SimHMRFormer',
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=False),
        input_dim=2048,
        with_bbox_info=True,
        num_joints=24,
        num_shape_query=1,
        num_cam_query=1,
        hidden_dim=256,
        position_encoding='sine',
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_49',
        model_path='data/body_models/smpl',
        keypoint_approximate=True,
        extra_joints_regressor='data/body_models/J_regressor_extra.npy'),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    convention='smpl_49',
    loss_keypoints3d=dict(type='MSELoss', loss_weight=400),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=300),
    # loss_vertex=dict(type='MSELoss', loss_weight=2.0),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=60),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=0.06))
    # loss_camera=dict(type='CameraPriorLoss', loss_weight=1))
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
# file_client_args = dict(backend='petrel', prefix='s3', path_mapping={'data/datasets':'s3://your-bucket/datasets'})
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    # dict(type='SyntheticOcclusion', occluders_file='data/occluders/pascal_occluders.npy'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_49'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    # dict(type='MixingErasing', ntype='self', mlist=['h36m']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=['img', *data_keys], meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'bbox_xywh',
            'ori_shape'])
            ]

test_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=file_client_args),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='ToTensor',
        keys=data_keys),
    dict(
        type='Collect',
        keys=[
            'img', *data_keys
        ],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'bbox_xywh',
            'ori_shape'
        ])
]
adv_data_keys = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_adv_pipeline = [
    dict(
        type='Collect',
        keys=adv_data_keys,
        meta_keys=[])
]
cache_files = dict(
    h36m='data/cache/h36m_mosh_train_smpl_49.npz',
    coco='data/cache/eft_coco_train_smpl_49.npz',
    lspet='data/cache/eft_lspet_train_smpl_49.npz',
    mpii='data/cache/eft_mpii_train_smpl_49.npz',
    mpi_inf_3dhp='data/cache/spin_mpi_inf_3dhp_train_smpl_49.npz')
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=16,
    train=dict(
        type='AdversarialDataset',
        train_dataset=dict(
            type='MixedDataset',
            configs=[
                dict(
                    type='HumanImageDataset',
                    dataset_name='h36m',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path='data/cache/h36m_mosh_train_smpl_49.npz',
                    convention='smpl_49',
                    ann_file='h36m_mosh_train.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='coco',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path='data/cache/eft_coco_train_smpl_49.npz',
                    convention='smpl_49',
                    ann_file='eft_coco_all.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='lspet',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path='data/cache/eft_lspet_train_smpl_49.npz',
                    convention='smpl_49',
                    ann_file='eft_lspet_dl.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='mpii',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path='data/cache/eft_mpii_train_smpl_49.npz',
                    convention='smpl_49',
                    ann_file='eft_mpii.npz'),
                dict(
                    type='HumanImageDataset',
                    dataset_name='mpi_inf_3dhp',
                    data_prefix='data',
                    pipeline=train_pipeline,
                    cache_data_path=
                    'data/cache/spin_mpi_inf_3dhp_train_smpl_49.npz',
                    convention='smpl_49',
                    ann_file='spin_mpi_inf_3dhp_train.npz')
            ],
            partition=[0.5, 0.233, 0.046, 0.021, 0.2],),
        adv_dataset=dict(
            type='MeshDataset',
            dataset_name='cmu_mosh',
            data_prefix='data',
            pipeline=train_adv_pipeline,
            ann_file='cmu_mosh.npz')),
    val=dict(
        type='HumanImageDataset',
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
        type='HumanImageDataset',
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz')
    )