from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import mmhuman3d.core.visualization.visualize_smpl as visualize_smpl
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx
from mmhuman3d.models.utils import FitsDict
from mmhuman3d.utils.geometry import (
    batch_rodrigues,
    estimate_translation,
    project_points,
    rotation_matrix_to_angle_axis,
)
from ..backbones.builder import build_backbone
from ..body_models.builder import build_body_model
from ..discriminators.builder import build_discriminator
from ..heads.builder import build_head
from ..losses.builder import build_loss
from ..necks.builder import build_neck
from ..registrants.builder import build_registrant
from .base_architecture import BaseArchitecture


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class BodyModelEstimator(BaseArchitecture, metaclass=ABCMeta):
    """BodyModelEstimator Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        neck (dict | None, optional): Neck config dict. Default: None
        head (dict | None, optional): Regressor config dict. Default: None.
        disc (dict | None, optional): Discriminator config dict.
            Default: None.
        registration (dict | None, optional): Registration config dict.
            Default: None.
        body_model_train (dict | None, optional): SMPL config dict during
            training. Default: None.
        body_model_test (dict | None, optional): SMPL config dict during
            test. Default: None.
        convention (str, optional): Keypoints convention. Default: "human_data"
        loss_keypoints2d (dict | None, optional): Losses config dict for
            2D keypoints. Default: None.
        loss_keypoints3d (dict | None, optional): Losses config dict for
            3D keypoints. Default: None.
        loss_vertex (dict | None, optional): Losses config dict for mesh
            vertices. Default: None
        loss_smpl_pose (dict | None, optional): Losses config dict for smpl
            pose. Default: None
        loss_smpl_betas (dict | None, optional): Losses config dict for smpl
            betas. Default: None
        loss_camera (dict | None, optional): Losses config dict for predicted
            camera parameters. Default: None
        loss_adv (dict | None, optional): Losses config for adversial
            training. Default: None.
        loss_segm_mask (dict | None, optional): Losses config for predicted
        part segmentation. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone: Optional[Union[dict, None]] = None,
                 neck: Optional[Union[dict, None]] = None,
                 head: Optional[Union[dict, None]] = None,
                 disc: Optional[Union[dict, None]] = None,
                 registration: Optional[Union[dict, None]] = None,
                 body_model_train: Optional[Union[dict, None]] = None,
                 body_model_test: Optional[Union[dict, None]] = None,
                 convention: Optional[str] = 'human_data',
                 loss_keypoints2d: Optional[Union[dict, None]] = None,
                 loss_keypoints3d: Optional[Union[dict, None]] = None,
                 loss_keypoints3d_model: Optional[Union[dict, None]] = None,
                 loss_vertex: Optional[Union[dict, None]] = None,
                 loss_smpl_pose: Optional[Union[dict, None]] = None,
                 loss_smpl_betas: Optional[Union[dict, None]] = None,
                 loss_camera: Optional[Union[dict, None]] = None,
                 loss_pose_prior: Optional[Union[dict, None]] = None,
                 loss_adv: Optional[Union[dict, None]] = None,
                 loss_segm_mask: Optional[Union[dict, None]] = None,
                 init_cfg: Optional[Union[list, dict, None]] = None):
        super(BodyModelEstimator, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.disc = build_discriminator(disc)

        self.body_model_train = build_body_model(body_model_train)
        self.body_model_test = build_body_model(body_model_test)
        self.convention = convention

        # TODO: support HMR+

        self.registration = registration
        if registration is not None:
            self.fits_dict = FitsDict(fits='static')
            self.registration_mode = self.registration['mode']
            self.registrant = build_registrant(registration['registrant'])
        else:
            self.registrant = None

        self.loss_keypoints2d = build_loss(loss_keypoints2d)
        self.loss_keypoints3d = build_loss(loss_keypoints3d)
        self.loss_keypoints3d_model = build_loss(loss_keypoints3d_model)

        self.loss_vertex = build_loss(loss_vertex)
        self.loss_smpl_pose = build_loss(loss_smpl_pose)
        self.loss_smpl_betas = build_loss(loss_smpl_betas)
        self.loss_adv = build_loss(loss_adv)
        self.loss_camera = build_loss(loss_camera)
        self.loss_pose_prior = build_loss(loss_pose_prior)
        self.loss_segm_mask = build_loss(loss_segm_mask)
        set_requires_grad(self.body_model_train, False)
        set_requires_grad(self.body_model_test, False)
        if self.loss_pose_prior is not None:
            set_requires_grad(self.loss_pose_prior, False)

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:
        1. get fake and real SMPL parameters
        2. optimize discriminator (if have)
        3. optimize generator
        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.
        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).
        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """
        if self.backbone is not None:
            img = data_batch['img']
            features = self.backbone(img)
        else:
            features = data_batch['features']

        if self.neck is not None:
            features = self.neck(features)

        predictions = self.head(features)
        targets = self.prepare_targets(data_batch)

        # optimize discriminator (if have)
        if self.disc is not None:
            self.optimize_discrinimator(predictions, data_batch, optimizer)

        if self.registration is not None:
            targets = self.run_registration(predictions, targets)

        losses = self.compute_losses(predictions, targets)
        # optimizer generator part
        if self.disc is not None:
            adv_loss = self.optimize_generator(predictions)
            losses.update(adv_loss)

        loss, log_vars = self._parse_losses(losses)
        for key in optimizer.keys():
            optimizer[key].zero_grad()
        loss.backward()
        for key in optimizer.keys():
            optimizer[key].step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs

    def run_registration(
            self,
            predictions: dict,
            targets: dict,
            threshold: Optional[float] = 10.0,
            focal_length: Optional[float] = 5000.0,
            img_res: Optional[Union[Tuple[int], int]] = 224) -> dict:
        """Run registration on 2D keypoinst in predictions to obtain SMPL
        parameters as pseudo ground truth.

        Args:
            predictions (dict): predicted SMPL parameters are used for
                initialization.
            targets (dict): existing ground truths with 2D keypoints
            threshold (float, optional): the threshold to update fits
                dictionary. Default: 10.0.
            focal_length (tuple(int) | int, optional): camera focal_length
            img_res (int, optional): image resolution

        Returns:
            targets: contains additional SMPL parameters
        """

        img_metas = targets['img_metas']
        dataset_name = [meta['dataset_name'] for meta in img_metas
                        ]  # name of the dataset the image comes from

        indices = targets['sample_idx'].squeeze()
        is_flipped = targets['is_flipped'].squeeze().bool(
        )  # flag that indicates whether image was flipped
        # during data augmentation
        rot_angle = targets['rotation'].squeeze(
        )  # rotation angle used for data augmentation Q
        gt_betas = targets['smpl_betas'].float()
        gt_global_orient = targets['smpl_global_orient'].float()
        gt_pose = targets['smpl_body_pose'].float().view(-1, 69)

        pred_rotmat = predictions['pred_pose'].detach().clone()
        pred_betas = predictions['pred_shape'].detach().clone()
        pred_cam = predictions['pred_cam'].detach().clone()
        pred_cam_t = torch.stack([
            pred_cam[:, 1], pred_cam[:, 2], 2 * focal_length /
            (img_res * pred_cam[:, 0] + 1e-9)
        ],
                                 dim=-1)

        gt_keypoints_2d = targets['keypoints2d'].float()
        num_keypoints = gt_keypoints_2d.shape[1]

        has_smpl = targets['has_smpl'].view(
            -1).bool()  # flag that indicates whether SMPL parameters are valid
        batch_size = has_smpl.shape[0]
        device = has_smpl.device

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as
        # it comes from SMPL
        gt_out = self.body_model_train(
            betas=gt_betas, body_pose=gt_pose, global_orient=gt_global_orient)
        # TODO: support more convention
        assert num_keypoints == 49
        gt_model_joints = gt_out['joints']
        gt_vertices = gt_out['vertices']

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(),
                                              rot_angle.cpu(),
                                              is_flipped.cpu())]

        opt_pose = opt_pose.to(device)
        opt_betas = opt_betas.to(device)
        opt_output = self.body_model_train(
            betas=opt_betas,
            body_pose=opt_pose[:, 3:],
            global_orient=opt_pose[:, :3])
        opt_joints = opt_output['joints']
        opt_vertices = opt_output['vertices']

        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints,
            gt_keypoints_2d_orig,
            focal_length=focal_length,
            img_size=img_res)

        opt_cam_t = estimate_translation(
            opt_joints,
            gt_keypoints_2d_orig,
            focal_length=focal_length,
            img_size=img_res)

        with torch.no_grad():
            loss_dict = self.registrant.evaluate(
                global_orient=opt_pose[:, :3],
                body_pose=opt_pose[:, 3:],
                betas=opt_betas,
                transl=opt_cam_t,
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                reduction_override='none')
        opt_joint_loss = loss_dict['keypoint2d_loss'].sum(dim=-1).sum(dim=-1)

        if self.registration_mode == 'in_the_loop':
            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([
                pred_rotmat.detach().view(-1, 3, 3).detach(),
                torch.tensor([0, 0, 1], dtype=torch.float32,
                             device=device).view(1, 3, 1).expand(
                                 batch_size * 24, -1, -1)
            ],
                                        dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(
                pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation,
            # so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            registrant_output = self.registrant(
                keypoints2d=gt_keypoints_2d_orig[:, :, :2],
                keypoints2d_conf=gt_keypoints_2d_orig[:, :, 2],
                init_global_orient=pred_pose[:, :3],
                init_transl=pred_cam_t,
                init_body_pose=pred_pose[:, 3:],
                init_betas=pred_betas,
                return_joints=True,
                return_verts=True,
                return_losses=True)
            new_opt_vertices = registrant_output[
                'vertices'] - pred_cam_t.unsqueeze(1)
            new_opt_joints = registrant_output[
                'joints'] - pred_cam_t.unsqueeze(1)

            new_opt_global_orient = registrant_output['global_orient']
            new_opt_body_pose = registrant_output['body_pose']
            new_opt_pose = torch.cat(
                [new_opt_global_orient, new_opt_body_pose], dim=1)

            new_opt_betas = registrant_output['betas']
            new_opt_cam_t = registrant_output['transl']
            new_opt_joint_loss = registrant_output['keypoint2d_loss'].sum(
                dim=-1).sum(dim=-1)

            # Will update the dictionary for the examples where the new loss
            # is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(),
                            is_flipped.cpu(),
                            update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters,
        # if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, 3:] = gt_pose[has_smpl, :]
        opt_pose[has_smpl, :3] = gt_global_orient[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with
        # the threshold
        valid_fit = (opt_joint_loss < threshold).to(device)
        valid_fit = valid_fit | has_smpl
        targets['valid_fit'] = valid_fit

        targets['opt_vertices'] = opt_vertices
        targets['opt_cam_t'] = opt_cam_t
        targets['opt_joints'] = opt_joints
        targets['opt_pose'] = opt_pose
        targets['opt_betas'] = opt_betas

        return targets

    def optimize_discrinimator(self, predictions: dict, data_batch: dict,
                               optimizer: dict):
        """Optimize discrinimator during adversarial training."""
        set_requires_grad(self.disc, True)
        fake_data = self.make_fake_data(predictions, requires_grad=False)
        real_data = self.make_real_data(data_batch)
        fake_score = self.disc(fake_data)
        real_score = self.disc(real_data)

        disc_losses = {}
        disc_losses['real_loss'] = self.loss_adv(
            real_score, target_is_real=True, is_disc=True)
        disc_losses['fake_loss'] = self.loss_adv(
            fake_score, target_is_real=False, is_disc=True)
        loss_disc, log_vars_d = self._parse_losses(disc_losses)

        optimizer['disc'].zero_grad()
        loss_disc.backward()
        optimizer['disc'].step()

    def optimize_generator(self, predictions: dict):
        """Optimize generator during adversarial training."""
        set_requires_grad(self.disc, False)
        fake_data = self.make_fake_data(predictions, requires_grad=True)
        pred_score = self.disc(fake_data)
        loss_adv = self.loss_adv(
            pred_score, target_is_real=True, is_disc=False)
        loss = dict(adv_loss=loss_adv)
        return loss

    def compute_keypoints3d_model_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d_model(
            pred_keypoints3d, gt_keypoints3d, reduction_override='none')
        # more xy
        # loss = loss * torch.tensor([2,2,1],device=loss.device)
        # If has_keypoints3d is not None, then computes the losses on the
        # instances that have ground-truth keypoints3d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints3d
        # which have positive confidence.

        # has_keypoints3d is None when the key has_keypoints3d
        # is not in the datasets
        if has_keypoints3d is None:

            valid_pos = keypoints3d_conf > 0
            if keypoints3d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = torch.sum(loss * keypoints3d_conf)
            loss /= keypoints3d_conf[valid_pos].numel()
        else:

            keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
            if keypoints3d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = loss[has_keypoints3d == 1]
            loss = (loss * keypoints3d_conf).mean()
        return loss
    
    def compute_keypoints3d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            gt_keypoints3d: torch.Tensor,
            has_keypoints3d: Optional[torch.Tensor] = None):
        """Compute loss for 3d keypoints."""
        keypoints3d_conf = gt_keypoints3d[:, :, 3].float().unsqueeze(-1)
        keypoints3d_conf = keypoints3d_conf.repeat(1, 1, 3)
        pred_keypoints3d = pred_keypoints3d.float()
        gt_keypoints3d = gt_keypoints3d[:, :, :3].float()

        # currently, only mpi_inf_3dhp and h36m have 3d keypoints
        # both datasets have right_hip_extra and left_hip_extra
        right_hip_idx = get_keypoint_idx('right_hip_extra', self.convention)
        left_hip_idx = get_keypoint_idx('left_hip_extra', self.convention)
        gt_pelvis = (gt_keypoints3d[:, right_hip_idx, :] +
                     gt_keypoints3d[:, left_hip_idx, :]) / 2
        pred_pelvis = (pred_keypoints3d[:, right_hip_idx, :] +
                       pred_keypoints3d[:, left_hip_idx, :]) / 2

        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
        loss = self.loss_keypoints3d(
            pred_keypoints3d, gt_keypoints3d, reduction_override='none')

        # If has_keypoints3d is not None, then computes the losses on the
        # instances that have ground-truth keypoints3d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints3d
        # which have positive confidence.

        # has_keypoints3d is None when the key has_keypoints3d
        # is not in the datasets
        if has_keypoints3d is None:

            valid_pos = keypoints3d_conf > 0
            if keypoints3d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = torch.sum(loss * keypoints3d_conf)
            loss /= keypoints3d_conf[valid_pos].numel()
        else:

            keypoints3d_conf = keypoints3d_conf[has_keypoints3d == 1]
            if keypoints3d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints3d)
            loss = loss[has_keypoints3d == 1]
            loss = (loss * keypoints3d_conf).mean()
        return loss

    def compute_keypoints2d_loss(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            img_res: Optional[int] = 224,
            focal_length: Optional[int] = 5000,
            has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()
        pred_keypoints2d = project_points(
            pred_keypoints3d,
            pred_cam,
            focal_length=focal_length,
            img_res=img_res)
        # Normalize keypoints to [-1,1]
        # The coordinate origin of pred_keypoints_2d is
        # the center of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1)
        # The coordinate origin of gt_keypoints_2d is
        # the top left corner of the input image.
        gt_keypoints2d = 2 * gt_keypoints2d / (img_res - 1) - 1
        loss = self.loss_keypoints2d(
            pred_keypoints2d, gt_keypoints2d, keypoints2d_conf, reduction_override='none')

        # If has_keypoints2d is not None, then computes the losses on the
        # instances that have ground-truth keypoints2d.
        # But the zero confidence keypoints will be included in mean.
        # Otherwise, only compute the keypoints2d
        # which have positive confidence.
        # has_keypoints2d is None when the key has_keypoints2d
        # is not in the datasets

        if has_keypoints2d is None:
            valid_pos = keypoints2d_conf > 0
            if keypoints2d_conf[valid_pos].numel() == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = torch.sum(loss * keypoints2d_conf)
            loss /= keypoints2d_conf[valid_pos].numel()
        else:
            keypoints2d_conf = keypoints2d_conf[has_keypoints2d == 1]
            if keypoints2d_conf.shape[0] == 0:
                return torch.Tensor([0]).type_as(gt_keypoints2d)
            loss = loss[has_keypoints2d == 1]
            loss = (loss * keypoints2d_conf).mean()

        return loss

    def compute_vertex_loss(self, pred_vertices: torch.Tensor,
                            gt_vertices: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for vertices."""
        gt_vertices = gt_vertices.float()
        conf = has_smpl.float().view(-1, 1, 1)
        conf = conf.repeat(1, gt_vertices.shape[1], gt_vertices.shape[2])
        loss = self.loss_vertex(
            pred_vertices, gt_vertices, reduction_override='none')
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_vertices)
        loss = torch.sum(loss * conf) / conf[valid_pos].numel()
        return loss

    def compute_smpl_pose_loss(self, pred_rotmat: torch.Tensor,
                               gt_pose: torch.Tensor, has_smpl: torch.Tensor):
        """Compute loss for smpl pose."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_pose)
        pred_rotmat = pred_rotmat[valid_pos]
        gt_pose = gt_pose[valid_pos]
        conf = conf[valid_pos]
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss = self.loss_smpl_pose(
            pred_rotmat, gt_rotmat, reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_smpl_betas_loss(self, pred_betas: torch.Tensor,
                                gt_betas: torch.Tensor,
                                has_smpl: torch.Tensor):
        """Compute loss for smpl betas."""
        conf = has_smpl.float().view(-1)
        valid_pos = conf > 0
        if conf[valid_pos].numel() == 0:
            return torch.Tensor([0]).type_as(gt_betas)
        pred_betas = pred_betas[valid_pos]
        gt_betas = gt_betas[valid_pos]
        conf = conf[valid_pos]
        loss = self.loss_smpl_betas(
            pred_betas, gt_betas, reduction_override='none')
        loss = loss.view(loss.shape[0], -1).mean(-1)
        loss = torch.mean(loss * conf)
        return loss

    def compute_camera_loss(self, cameras: torch.Tensor):
        """Compute loss for predicted camera parameters."""
        loss = self.loss_camera(cameras)
        return loss
    def compute_loss_pose_prior(self, poses: torch.Tensor):
        loss = self.loss_pose_prior(poses)
        return loss
    
    def compute_part_segmentation_loss(self,
                                       pred_heatmap: torch.Tensor,
                                       gt_vertices: torch.Tensor,
                                       gt_keypoints2d: torch.Tensor,
                                       gt_model_joints: torch.Tensor,
                                       has_smpl: torch.Tensor,
                                       img_res: Optional[int] = 224,
                                       focal_length: Optional[int] = 500):
        """Compute loss for part segmentations."""
        device = gt_keypoints2d.device
        gt_keypoints2d_valid = gt_keypoints2d[has_smpl == 1]
        batch_size = gt_keypoints2d_valid.shape[0]

        gt_vertices_valid = gt_vertices[has_smpl == 1]
        gt_model_joints_valid = gt_model_joints[has_smpl == 1]

        if batch_size == 0:
            return torch.Tensor([0]).type_as(gt_keypoints2d)
        gt_cam_t = estimate_translation(
            gt_model_joints_valid,
            gt_keypoints2d_valid,
            focal_length=focal_length,
            img_size=img_res,
        )

        K = torch.eye(3)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[2, 2] = 1
        K[0, 2] = img_res / 2.
        K[1, 2] = img_res / 2.
        K = K[None, :, :]

        R = torch.eye(3)[None, :, :]
        device = gt_keypoints2d.device
        gt_sem_mask = visualize_smpl.render_smpl(
            verts=gt_vertices_valid,
            R=R,
            K=K,
            T=gt_cam_t,
            render_choice='part_silhouette',
            resolution=img_res,
            return_tensor=True,
            body_model=self.body_model_train,
            device=device,
            in_ndc=False,
            convention='pytorch3d',
            projection='perspective',
            no_grad=True,
            batch_size=batch_size,
            verbose=False,
        )
        gt_sem_mask = torch.flip(gt_sem_mask, [1, 2]).squeeze(-1).detach()
        pred_heatmap_valid = pred_heatmap[has_smpl == 1]
        ph, pw = pred_heatmap_valid.size(2), pred_heatmap_valid.size(3)
        h, w = gt_sem_mask.size(1), gt_sem_mask.size(2)
        if ph != h or pw != w:
            pred_heatmap_valid = F.interpolate(
                input=pred_heatmap_valid, size=(h, w), mode='bilinear')

        loss = self.loss_segm_mask(pred_heatmap_valid, gt_sem_mask)
        return loss

    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_cam = predictions['pred_cam'].view(-1, 3)

        gt_keypoints3d = targets['keypoints3d'] # [bs, 49, 4]
        gt_keypoints2d = targets['keypoints2d'] # [bs, 49, 3]
        # pred_pose N, 24, 3, 3
        if self.body_model_train is not None:
            pred_output = self.body_model_train(
                betas=pred_betas,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False,
                num_joints=gt_keypoints2d.shape[1])
            pred_keypoints3d = pred_output['joints']
            pred_vertices = pred_output['vertices']

        # # TODO: temp. Should we multiply confs here?
        # pred_keypoints3d_mask = pred_output['joint_mask']
        # keypoints3d_mask = keypoints3d_mask * pred_keypoints3d_mask

        # TODO: temp solution
        if 'valid_fit' in targets:
            has_smpl = targets['valid_fit'].view(-1)
            # global_orient = targets['opt_pose'][:, :3].view(-1, 1, 3)
            gt_pose = targets['opt_pose']
            gt_betas = targets['opt_betas']
            gt_vertices = targets['opt_vertices']
        else:
            has_smpl = targets['has_smpl'].view(-1)
            gt_pose = targets['smpl_body_pose'] # [bs, num_joints, 3]
            global_orient = targets['smpl_global_orient'].view(-1, 1, 3) # [bs, 1, 3]
            gt_pose = torch.cat((global_orient, gt_pose), dim=1).float()
            gt_betas = targets['smpl_betas'].float()

            # gt_pose N, 72
            if self.body_model_train is not None:
                gt_output = self.body_model_train(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                    num_joints=gt_keypoints2d.shape[1])
                gt_vertices = gt_output['vertices']
                gt_model_joints = gt_output['joints']
        if 'has_keypoints3d' in targets:
            has_keypoints3d = targets['has_keypoints3d'].squeeze(-1)
        else:
            has_keypoints3d = None
        if 'has_keypoints2d' in targets:
            has_keypoints2d = targets['has_keypoints2d'].squeeze(-1)
        else:
            has_keypoints2d = None
        if 'pred_segm_mask' in predictions:
            pred_segm_mask = predictions['pred_segm_mask']
        losses = {}
        if self.loss_keypoints3d is not None:
            losses['keypoints3d_loss'] = self.compute_keypoints3d_loss(
                pred_keypoints3d,
                gt_keypoints3d,
                has_keypoints3d=has_keypoints3d)
        if self.loss_keypoints3d_model is not None:
            losses['keypoints3d_model_loss'] = self.compute_keypoints3d_model_loss(
                pred_keypoints3d,
                torch.cat((gt_model_joints,torch.ones((gt_model_joints.shape[0],gt_model_joints.shape[1],1),device=gt_model_joints.device)),dim=2),
                has_keypoints3d=torch.ones_like(has_keypoints3d))
        if self.loss_keypoints2d is not None:
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                has_keypoints2d=has_keypoints2d)
        if self.loss_vertex is not None:
            losses['vertex_loss'] = self.compute_vertex_loss(
                pred_vertices, gt_vertices, has_smpl)
        if self.loss_smpl_pose is not None:
            losses['smpl_pose_loss'] = self.compute_smpl_pose_loss(
                pred_pose, gt_pose, has_smpl)
        if self.loss_smpl_betas is not None:
            losses['smpl_betas_loss'] = self.compute_smpl_betas_loss(
                pred_betas, gt_betas, has_smpl)
        if self.loss_camera is not None:
            losses['camera_loss'] = self.compute_camera_loss(pred_cam)
        if self.loss_pose_prior is not None:
            losses['loss_pose_prior'] = self.compute_loss_pose_prior(pred_pose)
        if self.loss_segm_mask is not None:
            losses['loss_segm_mask'] = self.compute_part_segmentation_loss(
                pred_segm_mask, gt_vertices, gt_keypoints2d, gt_model_joints,
                has_smpl)

        return losses

    @abstractmethod
    def make_fake_data(self, predictions, requires_grad):
        pass

    @abstractmethod
    def make_real_data(self, data_batch):
        pass

    @abstractmethod
    def prepare_targets(self, data_batch):
        pass

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    @abstractmethod
    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        pass


class ImageBodyModelEstimator(BodyModelEstimator):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        pred_cam = predictions['pred_cam']
        pred_pose = predictions['pred_pose']
        pred_betas = predictions['pred_shape']
        if requires_grad:
            fake_data = (pred_cam, pred_pose, pred_betas)
        else:
            fake_data = (pred_cam.detach(), pred_pose.detach(),
                         pred_betas.detach())
        return fake_data

    def make_real_data(self, data_batch: dict):
        transl = data_batch['adv_smpl_transl'].float()
        global_orient = data_batch['adv_smpl_global_orient']
        body_pose = data_batch['adv_smpl_body_pose']
        betas = data_batch['adv_smpl_betas'].float()
        pose = torch.cat((global_orient, body_pose), dim=-1).float()
        real_data = (transl, pose, betas)
        return real_data

    def prepare_targets(self, data_batch: dict):
        # Image Mesh Estimator does not need extra process for ground truth
        return data_batch

    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)
        predictions = self.head(features)    
        pred_pose = predictions['pred_pose'] # list of [24, 3, 3] 
        pred_betas = predictions['pred_shape'] # list of [10]
        pred_cam = predictions['pred_cam']  # list of [3] for HMR
        pred_output = self.body_model_test(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        image_path,bbox_xywh,ori_shape = [],[],[]
        for img_meta in img_metas:
            image_path.append(img_meta['image_path'])
            if img_meta.get('bbox_xywh',None) is not None:
                bbox_xywh.append(img_meta['bbox_xywh'])
            if img_meta.get('ori_shape',None) is not None:
                ori_shape.append(img_meta['ori_shape'])
        all_preds['image_path'] = image_path
        all_preds['bbox_xywh'] = bbox_xywh
        all_preds['ori_shape'] = ori_shape
        all_preds['image_idx'] = kwargs['sample_idx']
        
        if kwargs.get("return_attn",False):
            all_preds['sa_weights'] = predictions['sa_weights'].detach().cpu().numpy() 
            all_preds['ca_weights'] = predictions['ca_weights'].detach().cpu().numpy() 
            all_preds['img'] = img.permute(0,2,3,1).contiguous().detach().cpu().numpy()
            all_preds['gt_keypoints2d'] = kwargs['keypoints2d'].detach().cpu().numpy()
        if kwargs.get("return_gt",False):
            all_preds['gt_smpl_body_pose'] = kwargs['smpl_body_pose'].detach().cpu().numpy()
            all_preds['img'] = img.permute(0,2,3,1).contiguous().detach().cpu().numpy()
            all_preds['gt_smpl_betas'] = kwargs['smpl_betas'].detach().cpu().numpy()
            all_preds['gt_smpl_transl'] = kwargs['smpl_transl'].detach().cpu().numpy()
            all_preds['gt_smpl_global_orient'] = kwargs['smpl_global_orient'].detach().cpu().numpy()
            all_preds['gt_keypoints2d'] = kwargs['keypoints2d'].detach().cpu().numpy()
            all_preds['img_h'] = kwargs['img_h'].detach().cpu().numpy()
            all_preds['img_w'] = kwargs['img_w'].detach().cpu().numpy()
            all_preds['center'] = kwargs['center'].detach().cpu().numpy()
            all_preds['scale'] = kwargs['scale'].detach().cpu().numpy()
            all_preds['crop_trans'] = kwargs['crop_trans'].detach().cpu().numpy()
            if kwargs.get('keypoints3d',None) is not None:
                all_preds['gt_keypoints3d'] = kwargs['keypoints3d'].detach().cpu().numpy()
            if kwargs.get('cam',None) is not None:
                all_preds['gt_cam'] = kwargs['cam'].detach().cpu().numpy()
            if kwargs.get('focal_length',None) is not None:
                all_preds['gt_focal_length'] = kwargs['focal_length'].detach().cpu().numpy()
            if kwargs.get('global_transl',None) is not None:
                all_preds['gt_global_transl'] = kwargs['global_transl'].detach().cpu().numpy()
            if kwargs.get('bbox_info',None) is not None:
                all_preds['gt_bbox_info'] = kwargs['bbox_info'].detach().cpu().numpy()
        return all_preds


class VideoBodyModelEstimator(BodyModelEstimator):

    def make_fake_data(self, predictions: dict, requires_grad: bool):
        B, T = predictions['pred_cam'].shape[:2]
        pred_cam_vec = predictions['pred_cam']
        pred_betas_vec = predictions['pred_shape']
        pred_pose = predictions['pred_pose']
        pred_pose_vec = rotation_matrix_to_angle_axis(pred_pose.view(-1, 3, 3))
        pred_pose_vec = pred_pose_vec.contiguous().view(B, T, -1)
        pred_theta_vec = (pred_cam_vec, pred_pose_vec, pred_betas_vec)
        pred_theta_vec = torch.cat(pred_theta_vec, dim=-1)

        if not requires_grad:
            pred_theta_vec = pred_theta_vec.detach()
        return pred_theta_vec[:, :, 6:75]

    def make_real_data(self, data_batch: dict):
        B, T = data_batch['adv_smpl_transl'].shape[:2]
        transl = data_batch['adv_smpl_transl'].view(B, T, -1)
        global_orient = \
            data_batch['adv_smpl_global_orient'].view(B, T, -1)
        body_pose = data_batch['adv_smpl_body_pose'].view(B, T, -1)
        betas = data_batch['adv_smpl_betas'].view(B, T, -1)
        real_data = (transl, global_orient, body_pose, betas)
        real_data = torch.cat(real_data, dim=-1).float()
        return real_data[:, :, 6:75]

    def prepare_targets(self, data_batch: dict):
        # Video Mesh Estimator needs squeeze first two dimensions
        B, T = data_batch['smpl_body_pose'].shape[:2]

        output = {
            'smpl_body_pose': data_batch['smpl_body_pose'].view(-1, 23, 3),
            'smpl_global_orient': data_batch['smpl_global_orient'].view(-1, 3),
            'smpl_betas': data_batch['smpl_betas'].view(-1, 10),
            'has_smpl': data_batch['has_smpl'].view(-1),
            'keypoints3d': data_batch['keypoints3d'].view(B * T, -1, 4),
            'keypoints2d': data_batch['keypoints2d'].view(B * T, -1, 3)
        }
        return output

    def forward_test(self, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(kwargs['img'])
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)

        B, T = features.shape[:2]
        predictions = self.head(features)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_cam = predictions['pred_cam'].view(-1, 3)

        pred_output = self.body_model_test(
            betas=pred_betas,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, 0].unsqueeze(1),
            pose2rot=False)

        pred_vertices = pred_output['vertices']
        pred_keypoints_3d = pred_output['joints']
        all_preds = {}
        all_preds['keypoints_3d'] = pred_keypoints_3d.detach().cpu().numpy()
        all_preds['smpl_pose'] = pred_pose.detach().cpu().numpy()
        all_preds['smpl_beta'] = pred_betas.detach().cpu().numpy()
        all_preds['camera'] = pred_cam.detach().cpu().numpy()
        all_preds['vertices'] = pred_vertices.detach().cpu().numpy()
        all_preds['image_idx'] = \
            kwargs['sample_idx'].detach().cpu().numpy().reshape((-1))
        return all_preds
