import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
    get_mapping,
)
from mmhuman3d.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)
from mmhuman3d.models.body_models.builder import build_body_model
from .base_dataset import BaseDataset
from .builder import DATASETS
from plyfile import PlyData, PlyElement

def save_ply(xyz, path=None):
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    attributes = [xyz]
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    

@DATASETS.register_module()
class HumanImageDataset(BaseDataset, metaclass=ABCMeta):
    """Human Image Dataset.

    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmhuman3d.datasets.pipelines`.
        dataset_name (str | None): the name of dataset. It is used to
            identify the type of evaluation metric. Default: None.
        body_model (dict | None, optional): the config for body model,
            which will be used to generate meshes and keypoints.
            Default: None.
        ann_file (str | None, optional): the annotation file. When ann_file
            is str, the subclass is expected to read from the ann_file.
            When ann_file is None, the subclass is expected to read
            according to data_prefix.
        convention (str, optional): keypoints convention. Keypoints will be
            converted from "human_data" to the given one.
            Default: "human_data"
        cache_data_path (str | None, optional): the path to store the cache
            file. When cache_data_path is None, each dataset will store a copy
            into memory. If cache_data_path is set, the dataset will first
            create one cache file and then use a cache reader to reduce memory
            cost and initialization time. The cache file will be generated
            only once if they are not found at the the path. Otherwise, only
            cache readers will be established.
        test_mode (bool, optional): in train mode or test mode.
            Default: False.
    """
    # metric
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc',
        'ihmr'
    }

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: str,
                 body_model: Optional[Union[dict, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 convention: Optional[str] = 'human_data',
                 cache_data_path: Optional[Union[str, None]] = None,
                 test_mode: Optional[bool] = False):
        self.convention = convention
        self.num_keypoints = get_keypoint_num(convention)
        self.cache_data_path = cache_data_path
        super(HumanImageDataset,
              self).__init__(data_prefix, pipeline, ann_file, test_mode,
                             dataset_name)
        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None

    def get_annotation_file(self):
        """Get path of the annotation file."""
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    def load_annotations(self):
        """Load annotation from the annotation file.

        Here we simply use :obj:`HumanData` to parse the annotation.
        """
        rank, world_size = get_dist_info()
        self.get_annotation_file()
        if self.cache_data_path is None:
            use_human_data = True
        elif rank == 0 and not os.path.exists(self.cache_data_path):
            use_human_data = True
        else:
            use_human_data = False
        if use_human_data:
            self.human_data = HumanData.fromfile(self.ann_file)

            if self.human_data.check_keypoints_compressed():
                self.human_data.decompress_keypoints()
            # change keypoint from 'human_data' to the given convention
            if 'keypoints3d' in self.human_data:
                keypoints3d = self.human_data['keypoints3d']
                assert 'keypoints3d_mask' in self.human_data
                keypoints3d_mask = self.human_data['keypoints3d_mask']
                keypoints3d, keypoints3d_mask = \
                    convert_kps(
                        keypoints3d,
                        src='human_data',
                        dst=self.convention,
                        mask=keypoints3d_mask)
                self.human_data.__setitem__('keypoints3d', keypoints3d)
                self.human_data.__setitem__('keypoints3d_convention',
                                            self.convention)
                self.human_data.__setitem__('keypoints3d_mask',
                                            keypoints3d_mask)
            if 'keypoints2d' in self.human_data:
                keypoints2d = self.human_data['keypoints2d']
                assert 'keypoints2d_mask' in self.human_data
                keypoints2d_mask = self.human_data['keypoints2d_mask']
                keypoints2d, keypoints2d_mask = \
                    convert_kps(
                        keypoints2d,
                        src='human_data',
                        dst=self.convention,
                        mask=keypoints2d_mask)
                self.human_data.__setitem__('keypoints2d', keypoints2d)
                self.human_data.__setitem__('keypoints2d_convention',
                                            self.convention)
                self.human_data.__setitem__('keypoints2d_mask',
                                            keypoints2d_mask)
            self.human_data.compress_keypoints_by_mask()

        if self.cache_data_path is not None:
            if rank == 0 and not os.path.exists(self.cache_data_path):
                writer_kwargs, sliced_data = self.human_data.get_sliced_cache()
                writer = HumanDataCacheWriter(**writer_kwargs)
                writer.update_sliced_dict(sliced_data)
                writer.dump(self.cache_data_path)
            if world_size > 1:
                dist.barrier()
            self.cache_reader = HumanDataCacheReader(
                npz_path=self.cache_data_path)
            self.num_data = self.cache_reader.data_len
            self.human_data = None
        else:
            self.cache_reader = None
            self.num_data = self.human_data.data_len

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        sample_idx = idx
        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]
        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)
        if image_path.endswith('smc'):
            device, device_id, frame_id = self.human_data['image_id'][idx]
            info['image_id'] = (device, int(device_id), int(frame_id))

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx
        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, s = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
            info['has_keypoints2d'] = 1
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
            info['has_keypoints2d'] = 0
        if 'keypoints3d' in self.human_data:
            info['keypoints3d'] = self.human_data['keypoints3d'][idx]
            info['has_keypoints3d'] = 1
        else:
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
            info['has_keypoints3d'] = 0

        if 'cam' in self.human_data:
            info['cam'] = self.human_data['cam'][idx]
        elif 'cam_param' in self.human_data:
            try:
                cam_item = self.human_data['cam_param'][idx]
            except:
                cam_item = self.human_data['cam_param'][(str(image_path).split("/")[0],str(image_path).split("/")[2].split(".")[-1])]
            info['cam'] = np.concatenate((np.array(cam_item['in_mat']).reshape(-1),np.array(cam_item['rotation_mat']).reshape(-1),np.array(cam_item['translation']).reshape(-1)))
        
        if 'global_transl' in self.human_data:
            info['global_transl'] = self.human_data['global_transl'][idx]
        if 'focal_length' in self.human_data:
            info['focal_length'] = self.human_data['focal_length'][idx]

        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}

        if 'smpl' in self.human_data:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = int(self.human_data['has_smpl'][idx])
            else:
                info['has_smpl'] = 1
        else:
            info['has_smpl'] = 0
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][idx]
        else:
            info['smpl_body_pose'] = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][idx]
        else:
            info['smpl_global_orient'] = np.zeros((3))

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][idx]
        else:
            info['smpl_betas'] = np.zeros((10))

        if 'trans' in smpl_dict:
            info['smpl_transl'] = smpl_dict['trans'][idx]
        elif 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][idx]
        else:
            info['smpl_transl'] = np.zeros((3))

        return info

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        return self.pipeline(info)

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        # for keeping correctness during multi-gpu test, we sort all results

        res_dict = {}
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['keypoints_3d'])
            for i in range(batch_size):
                res_dict[int(target_id[i]) - 179] = dict(
                    keypoints=out['keypoints_3d'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                )

        keypoints, poses, betas = [], [], []

        # FOR DEBUG ONLY
        self.num_data = 1

        for i in range(self.num_data):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])

        res = dict(keypoints=keypoints, poses=poses, betas=betas)
        mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples, error = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples, error = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples, error = self._report_pve(res)
            elif _metric == 'ihmr':
                _nv_tuples = self._report_ihmr(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value, error

    @staticmethod
    def _write_keypoint_results(keypoints: Any, res_file: str):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _parse_result(self, res, mode='keypoint', body_part=None):
        """Parse results."""

        if mode == 'vertice':
            # gt
            gt_beta, gt_pose, gt_global_orient, gender = [], [], [], []
            gt_smpl_dict = self.human_data['smpl']
            for idx in range(self.num_data):
                gt_beta.append(gt_smpl_dict['betas'][idx])
                gt_pose.append(gt_smpl_dict['body_pose'][idx])
                gt_global_orient.append(gt_smpl_dict['global_orient'][idx])
                gender.append(0)
                # if self.human_data['meta']['gender'][idx] == 'm':
                #     gender.append(0)
                # else:
                #     gender.append(1)
            gt_beta = torch.FloatTensor(gt_beta)
            gt_pose = torch.FloatTensor(gt_pose).view(-1, 69)
            gt_global_orient = torch.FloatTensor(gt_global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(
                betas=gt_beta,
                body_pose=gt_pose,
                global_orient=gt_global_orient,
                gender=gender)
            gt_vertices = gt_output['vertices'].detach().cpu().numpy() * 1000.
            gt_mask = np.ones(gt_vertices.shape[:-1])
            # pred
            pred_pose = torch.FloatTensor(res['poses'])
            pred_beta = torch.FloatTensor(res['betas'])
            pred_output = self.body_model(
                betas=pred_beta,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False,
                gender=gender)
            pred_vertices = pred_output['vertices'].detach().cpu().numpy(
            ) * 1000.

            assert len(pred_vertices) == self.num_data
            # save_ply(pred_vertices[2], path="pred_vertices.ply")
            # save_ply(gt_vertices[2], path="gt_vertices.ply")
            return pred_vertices, gt_vertices, gt_mask
        elif mode == 'keypoint':
            pred_keypoints3d = res['keypoints']
            assert len(pred_keypoints3d) == self.num_data
            # (B, 17, 3)
            pred_keypoints3d = np.array(pred_keypoints3d)

            if self.dataset_name == 'pw3d':
                betas = [] 
                body_pose = []
                global_orient = []
                gender = []
                smpl_dict = self.human_data['smpl']
                for idx in range(self.num_data):
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                    if self.human_data['meta']['gender'][idx] == 'm':
                        gender.append(0)
                    else:
                        gender.append(1)
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gender = torch.Tensor(gender)
                gt_output = self.body_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    gender=gender)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
            elif self.dataset_name == 'h36m': 
                if "smpl" in self.human_data.keys():
                    betas = [] 
                    body_pose = []
                    global_orient = []
                    smpl_dict = self.human_data['smpl']
                    for idx in range(self.num_data):
                        betas.append(smpl_dict['betas'][idx])
                        body_pose.append(smpl_dict['body_pose'][idx])
                        global_orient.append(smpl_dict['global_orient'][idx])
                            
                    betas = torch.FloatTensor(betas)
                    body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                    global_orient = torch.FloatTensor(global_orient)
                    gt_output = self.body_model(
                        betas=betas,
                        body_pose=body_pose,
                        global_orient=global_orient,
                        gender=torch.zeros(self.num_data))
                    gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                    gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 17))
                    
                    # _, h36m_idxs, _ = get_mapping('human_data', 'h36m')
                    # anno_kpt = self.human_data['keypoints3d'][:self.num_data, h36m_idxs, :3]
                    # anno_kpt = anno_kpt[:self.num_data, :, :]

                else:
                    _, h36m_idxs, _ = get_mapping('human_data', 'h36m')
                    gt_keypoints3d = \
                        self.human_data['keypoints3d'][:self.num_data, h36m_idxs, :3]
                    gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 17))
            #if self.dataset_name == 'mpi_inf_3dhp':
            elif self.dataset_name == 'mpi_inf_3dhp':
                _, hp3d_idxs, _ = get_mapping('human_data',
                                              'mpi_inf_3dhp_test')
                gt_keypoints3d = self.human_data['keypoints3d'][:,hp3d_idxs,:3]
                joint_mapper = [
                    14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4
                ]
                gt_keypoints3d = gt_keypoints3d[:self.num_data, joint_mapper]
                gt_keypoints3d_mask = np.ones(
                    (len(gt_keypoints3d), len(joint_mapper)))
            elif self.dataset_name == 'humman':
                betas = []
                body_pose = []
                global_orient = []
                smpl_dict = self.human_data['smpl']
                for idx in range(self.num_data):
                    betas.append(smpl_dict['betas'][idx])
                    body_pose.append(smpl_dict['body_pose'][idx])
                    global_orient.append(smpl_dict['global_orient'][idx])
                betas = torch.FloatTensor(betas)
                body_pose = torch.FloatTensor(body_pose).view(-1, 69)
                global_orient = torch.FloatTensor(global_orient)
                gt_output = self.body_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient)
                gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
                gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
            else:
                raise NotImplementedError()

            # SMPL_49 only!
            if gt_keypoints3d.shape[1] == 49:
                assert pred_keypoints3d.shape[1] == 49

                gt_keypoints3d = gt_keypoints3d[:, 25:, :]
                pred_keypoints3d = pred_keypoints3d[:, 25:, :]

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            # H36M for testing!
            elif gt_keypoints3d.shape[1] == 17:
                assert pred_keypoints3d.shape[1] == 17
                # H36M_CORRECT_TO_H36M = [0,4,5,6,1,2,3,7,8,9,10,11,12,13,14,15,16]
                # pred_keypoints3d = pred_keypoints3d[:, H36M_CORRECT_TO_H36M, :]
                H36M_TO_J17 = [
                    6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
                ]
                H36M_TO_J14 = H36M_TO_J17[:14]
                joint_mapper = H36M_TO_J14

                pred_pelvis = pred_keypoints3d[:, 0]
                gt_pelvis = gt_keypoints3d[:, 0]
                #anno_pelvis = anno_kpt[:, 0]

                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]
                #anno_kpt = anno_kpt[:, joint_mapper, :]

            # keypoint 24
            elif gt_keypoints3d.shape[1] == 24:
                assert pred_keypoints3d.shape[1] == 24

                joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

                # we only evaluate on 14 lsp joints
                pred_pelvis = (pred_keypoints3d[:, 2] +
                               pred_keypoints3d[:, 3]) / 2
                gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

            else:
                pass

            pred_keypoints3d = (pred_keypoints3d - pred_pelvis[:, None, :]) * 1000
            gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000
            # anno_kpt = (anno_kpt - anno_pelvis[:, None, :]) * 1000

            gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

    def _report_mpjpe(self, res_file, metric='mpjpe', body_part=''):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint', body_part=body_part)

        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        if metric == 'mpjpe':
            alignment = 'none'
        elif metric == 'pa-mpjpe':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_mpjpe(pred_keypoints3d, gt_keypoints3d,
                               gt_keypoints3d_mask, alignment)
        info_str = [(err_name, error.mean())]
        # ((pred_keypoints3d - gt_keypoints3d) ** 2).mean(0).mean(0) ** 0.5
        return info_str, error

    def _report_3d_pck(self, res_file, metric='3dpck'):
        """Cauculate Percentage of Correct Keypoints (3DPCK) w. or w/o
        Procrustes alignment.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dpck'``: Standard 3DPCK.
                - ``'pa-3dpck'``:
                    3DPCK after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file)

        err_name = metric.upper()
        if metric == '3dpck':
            alignment = 'none'
        elif metric == 'pa-3dpck':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_pck(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_3d_auc(self, res_file, metric='3dauc'):
        """Cauculate the Area Under the Curve (AUC) computed for a range of
        3DPCK thresholds.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dauc'``: Standard 3DAUC.
                - ``'pa-3dauc'``: 3DAUC after aligning prediction to
                    groundtruth via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file)

        err_name = metric.upper()
        if metric == '3dauc':
            alignment = 'none'
        elif metric == 'pa-3dauc':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_auc(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_pve(self, res_file, metric='pve', body_part=''):
        """Cauculate per vertex error."""
        pred_verts, gt_verts, _ = \
            self._parse_result(res_file, mode='vertice', body_part=body_part)
        err_name = metric.upper()
        if body_part != '':
            err_name = body_part.upper() + ' ' + err_name

        if metric == 'pve':
            alignment = 'none'
        elif metric == 'pa-pve':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')
        error = vertice_pve(pred_verts, gt_verts, alignment)
        return [(err_name, error)], error

    def _report_ihmr(self, res_file):
        """Calculate IHMR metric.

        https://arxiv.org/abs/2203.16427
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint')

        pred_verts, gt_verts, _ = \
            self._parse_result(res_file, mode='vertice')

        from mmhuman3d.utils.geometry import rot6d_to_rotmat
        mean_param_path = 'data/body_models/smpl_mean_params.npz'
        mean_params = np.load(mean_param_path)
        mean_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        mean_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        mean_pose = rot6d_to_rotmat(mean_pose).view(1, 24, 3, 3)
        mean_output = self.body_model(
            betas=mean_shape,
            body_pose=mean_pose[:, 1:],
            global_orient=mean_pose[:, :1],
            pose2rot=False)
        mean_verts = mean_output['vertices'].detach().cpu().numpy() * 1000.
        dis = (gt_verts - mean_verts) * (gt_verts - mean_verts)
        dis = np.sqrt(dis.sum(axis=-1)).mean(axis=-1)
        # from the most remote one to the nearest one
        idx_order = np.argsort(dis)[::-1]
        num_data = idx_order.shape[0]

        def report_ihmr_idx(idx):
            mpvpe = vertice_pve(pred_verts[idx], gt_verts[idx])
            mpjpe = keypoint_mpjpe(pred_keypoints3d[idx], gt_keypoints3d[idx],
                                   gt_keypoints3d_mask[idx], 'none')
            pampjpe = keypoint_mpjpe(pred_keypoints3d[idx],
                                     gt_keypoints3d[idx],
                                     gt_keypoints3d_mask[idx], 'procrustes')
            return (mpvpe, mpjpe, pampjpe)

        def report_ihmr_tail(percentage):
            cur_data = int(num_data * percentage / 100.0)
            idx = idx_order[:cur_data]
            mpvpe, mpjpe, pampjpe = report_ihmr_idx(idx)
            res_mpvpe = ('bMPVPE Tail ' + str(percentage) + '%', mpvpe)
            res_mpjpe = ('bMPJPE Tail ' + str(percentage) + '%', mpjpe)
            res_pampjpe = ('bPA-MPJPE Tail ' + str(percentage) + '%', pampjpe)
            return [res_mpvpe, res_mpjpe, res_pampjpe]

        def report_ihmr_all(num_bin):
            num_per_bin = np.array([0 for _ in range(num_bin)
                                    ]).astype(np.float32)
            sum_mpvpe = np.array([0
                                  for _ in range(num_bin)]).astype(np.float32)
            sum_mpjpe = np.array([0
                                  for _ in range(num_bin)]).astype(np.float32)
            sum_pampjpe = np.array([0 for _ in range(num_bin)
                                    ]).astype(np.float32)
            max_dis = dis[idx_order[0]]
            min_dis = dis[idx_order[-1]]
            delta = (max_dis - min_dis) / num_bin
            for i in range(num_data):
                idx = int((dis[i] - min_dis) / delta - 0.001)
                res_mpvpe, res_mpjpe, res_pampjpe = report_ihmr_idx([i])
                num_per_bin[idx] += 1
                sum_mpvpe[idx] += res_mpvpe
                sum_mpjpe[idx] += res_mpjpe
                sum_pampjpe[idx] += res_pampjpe
            for i in range(num_bin):
                if num_per_bin[i] > 0:
                    sum_mpvpe[i] = sum_mpvpe[i] / num_per_bin[i]
                    sum_mpjpe[i] = sum_mpjpe[i] / num_per_bin[i]
                    sum_pampjpe[i] = sum_pampjpe[i] / num_per_bin[i]
            valid_idx = np.where(num_per_bin > 0)
            res_mpvpe = ('bMPVPE All', sum_mpvpe[valid_idx].mean())
            res_mpjpe = ('bMPJPE All', sum_mpjpe[valid_idx].mean())
            res_pampjpe = ('bPA-MPJPE All', sum_pampjpe[valid_idx].mean())
            return [res_mpvpe, res_mpjpe, res_pampjpe]

        metrics = []
        metrics.extend(report_ihmr_all(num_bin=100))
        metrics.extend(report_ihmr_tail(percentage=10))
        metrics.extend(report_ihmr_tail(percentage=5))
        return metrics
