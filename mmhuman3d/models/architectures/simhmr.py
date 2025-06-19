from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import mmhuman3d.core.visualization.visualize_smpl as visualize_smpl
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx
from mmhuman3d.models.utils import FitsDict
from mmhuman3d.models.architectures.mesh_estimator import BodyModelEstimator
from mmhuman3d.utils.geometry import (
    perspective_projection,
    cam_crop2full,
)

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

'''
A HMR model built with DETR
'''
class SimHMR(BodyModelEstimator):

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
    
    def compute_losses(self, predictions: dict, targets: dict):
        """Compute losses."""
        pred_betas = predictions['pred_shape'].view(-1, 10)
        pred_pose = predictions['pred_pose'].view(-1, 24, 3, 3)
        pred_cam_crop = predictions['pred_cam'].view(-1, 3)

        # NOTE: convert cam parameters from the crop to the full camera
        img_h, img_w = targets['img_h'], targets['img_w']
        center, scale, focal_length = targets['center'], targets[
            'scale'][:, 0], targets['focal_length'].squeeze(dim=1)
        full_img_shape = torch.hstack((img_h, img_w))
        pred_cam = cam_crop2full(pred_cam_crop, center, scale, full_img_shape,
                                 focal_length).to(torch.float32)

        gt_keypoints3d = targets['keypoints3d']
        # this should be in full frame
        gt_keypoints2d = targets['keypoints2d']
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

        # NOTE: use crop_trans to contain full -> crop so that pred keypoints
        # are normalized to bbox
        camera_center = torch.hstack((img_w, img_h)) / 2
        trans = targets['crop_trans'].float()

        # TODO: temp solution
        if 'valid_fit' in targets:
            has_smpl = targets['valid_fit'].view(-1)
            # global_orient = targets['opt_pose'][:, :3].view(-1, 1, 3)
            gt_pose = targets['opt_pose']
            gt_betas = targets['opt_betas']
            gt_vertices = targets['opt_vertices']
        else:
            has_smpl = targets['has_smpl'].view(-1)
            gt_pose = targets['smpl_body_pose']
            global_orient = targets['smpl_global_orient'].view(-1, 1, 3)
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
            losses['keypoints2d_loss'] = self.compute_keypoints2d_loss_cliff(
                pred_keypoints3d,
                pred_cam,
                gt_keypoints2d,
                camera_center,
                focal_length,
                trans,
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
        if self.loss_segm_mask is not None:
            losses['loss_segm_mask'] = self.compute_part_segmentation_loss(
                pred_segm_mask, gt_vertices, gt_keypoints2d, gt_model_joints,
                has_smpl)

        return losses
    
    def compute_keypoints2d_loss_cliff(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            camera_center: torch.Tensor,
            focal_length: torch.Tensor,
            trans: torch.Tensor,
            img_res: Optional[int] = 224,
            has_keypoints2d: Optional[torch.Tensor] = None):
        """Compute loss for 2d keypoints."""
        keypoints2d_conf = gt_keypoints2d[:, :, 2].float().unsqueeze(-1)
        keypoints2d_conf = keypoints2d_conf.repeat(1, 1, 2)
        gt_keypoints2d = gt_keypoints2d[:, :, :2].float()

        device = gt_keypoints2d.device
        batch_size = pred_keypoints3d.shape[0]

        pred_keypoints2d = perspective_projection(
            pred_keypoints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(
                batch_size, -1, -1),
            translation=pred_cam,
            focal_length=focal_length,
            camera_center=camera_center)

        pred_keypoints2d = torch.cat(
            (pred_keypoints2d, torch.ones(batch_size, pred_keypoints2d.shape[1], 1).to(device)),
            dim=2)
        # trans @ pred_keypoints2d2
        pred_keypoints2d = torch.einsum('bij,bkj->bki', trans,
                                        pred_keypoints2d)

        # The coordinate origin of pred_keypoints_2d and gt_keypoints_2d is
        # the top left corner of the input image.
        pred_keypoints2d = 2 * pred_keypoints2d / (img_res - 1) - 1
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
    
    def forward_test(self, img: torch.Tensor, img_metas: dict, **kwargs):
        """Defines the computation performed at every call when testing."""
        if self.backbone is not None:
            features = self.backbone(img)
        else:
            features = kwargs['features']

        if self.neck is not None:
            features = self.neck(features)
        # NOTE: extras for Cliff inference
        bbox_info = kwargs['bbox_info']
        predictions = self.head(features,bbox_info)    
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
        
        if kwargs.get("return_attn2",False):
            all_preds['sa_weights'] = predictions['sa_attn']
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

        # NOTE: features and bbox_info taken as input for Cliff
        bbox_info = data_batch['bbox_info']
        predictions = self.head(features, bbox_info)
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

    def optimize_discrinimator(self, predictions: dict, data_batch: dict,
                               optimizer: dict):
        """Optimize discrinimator during adversarial training."""
        set_requires_grad(self.disc, True)
        fake_data = self.make_fake_data(predictions, requires_grad=False)
        real_data = self.make_real_data(data_batch)
        fake_score = self.disc(fake_data)
        real_score = self.disc(real_data)

        # disc_losses = {}
        # disc_losses['real_loss'] = self.loss_adv(
        #     real_score, target_is_real=True, is_disc=True)
        # disc_losses['fake_loss'] = self.loss_adv(
        #     fake_score, target_is_real=False, is_disc=True)
        
        disc_losses = {}
        real_loss = self.loss_adv(
            real_score, target_is_real=True, is_disc=True)
        fake_loss = self.loss_adv(
            fake_score, target_is_real=False, is_disc=True)
        
        if real_loss.isnan() or fake_loss.isnan():
            disc_losses['fake_loss'] = 0.0
            disc_losses['real_loss'] = 0.0
            return
        else:
            disc_losses['fake_loss'] = fake_loss
            disc_losses['real_loss'] = real_loss

        loss_disc, log_vars_d = self._parse_losses(disc_losses)

        optimizer['disc'].zero_grad()
        loss_disc.backward()
        optimizer['disc'].step()