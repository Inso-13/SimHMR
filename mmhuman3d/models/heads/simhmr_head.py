'''
Replace the MLP head with a SA head.
'''
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule
from mmhuman3d.utils.geometry import rot6d_to_rotmat

from ..utils.builder import build_transformer

class MLP(nn.Module):
    def __init__(self, 
                 input_dim=128,
                 hidden_dim=128,
                 out_dim=1,
                 depth=2):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dim),
                       nn.ReLU()]
        
        for i in range(depth-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)
           

class PoseMLP(nn.Module):
    def __init__(self, 
                 num_joints=24,
                 hidden_dim=256,
                 out_dim=6):
        super().__init__()
        self.num_joints = num_joints
        #self.out_layers = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for i in range(num_joints)])
        self.out_layers = nn.ModuleList([MLP(input_dim=hidden_dim, out_dim=6) for i in range(num_joints)])
    
    def forward(self, pose_feat):
        # pose_feat： [bs, num_joints, hidden_dim]
        pose_out = [self.out_layers[i](pose_feat[:,i,:].unsqueeze(1)) for i in range(self.num_joints)]

        return torch.cat(pose_out, dim=1) # [bs, num_joints, 6]

class SelfAttentionPoseHead(nn.Module):
    def __init__(self, 
                 num_joints=24,
                 hidden_dim=256,
                 nhead=1,
                 dropout=0.0,
                 out_dim=6,
                 with_bbox=True):
        super().__init__()
        self.num_joints = num_joints
        
        if with_bbox:
            self.self_attn = nn.MultiheadAttention(hidden_dim+3, nhead, dropout=dropout)
            self.pose_ffn = nn.ModuleList([MLP(input_dim=hidden_dim+3, out_dim=out_dim) for i in range(num_joints)])
            self.norm1 = nn.LayerNorm(hidden_dim + 3)
        else:
            self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
            self.pose_ffn = nn.ModuleList([MLP(input_dim=hidden_dim, out_dim=out_dim) for i in range(num_joints)])
            self.norm1 = nn.LayerNorm(hidden_dim)

        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        

    
    def forward(self, pose_feat, bbox_info):
        if bbox_info is not None:
            bbox_info = bbox_info.unsqueeze(1).repeat(1, pose_feat.shape[1], 1) # [bs, J, 3]bo
            pose_feat = torch.cat([pose_feat, bbox_info], dim=-1) # [bs, J, dim + 3]

        pose_feat = pose_feat.permute(1,0,2)
        src2 = self.self_attn(pose_feat, pose_feat, value=pose_feat)[0]
        src = pose_feat + self.dropout(src2)
        src = self.norm1(src)

        pose_out = [self.pose_ffn[i](src[i,:,:].unsqueeze(1)) for i in range(self.num_joints)]
        pred_pose = torch.cat(pose_out, dim=1)

        return pred_pose



class SimHMRHead(BaseModule):

    def __init__(self,
                 transformer,
                 input_dim=2048,
                 num_joints=24, # include the root joint
                 num_shape_query=1,
                 num_cam_query=1,
                 hidden_dim=1024,
                 smpl_mean_params=None,
                 position_encoding="sine",
                 with_bbox_info=False,
                 init_cfg=None):
        super(SimHMRHead, self).__init__(init_cfg=init_cfg)
        
        self.transformer = build_transformer(transformer)
        self.position_embedding = position_encoding
        self.num_joints = num_joints
        self.with_bbox_info = with_bbox_info
        self.input_proj = nn.Conv2d(in_channels=input_dim, 
                                    out_channels=hidden_dim, 
                                    kernel_size=1, 
                                    stride=1)       
        
        self.position_embedding = build_position_encoding(position_encoding, hidden_dim)

        # define the query embedding for pose, shape, and camera parameters.
        self.pose_query = nn.Embedding(num_joints, hidden_dim)
        self.shape_query = nn.Embedding(num_shape_query, hidden_dim)
        self.cam_query = nn.Embedding(num_cam_query, hidden_dim)

        # prediction heads that take query vectors as inputs
        # self.pose_head = PoseMLP(num_joints=num_joints - 1, # excluding the root joint
        #                          hidden_dim=hidden_dim,
        #                          out_dim=6)
        

        self.pose_head = SelfAttentionPoseHead(num_joints=num_joints,
                                               hidden_dim=hidden_dim,
                                               out_dim=6,
                                               with_bbox=with_bbox_info)
        
        self.shape_head = MLP(input_dim=num_shape_query * hidden_dim, out_dim=10)
        self.cam_head = MLP(input_dim=num_cam_query * hidden_dim, out_dim=3)

        # initial smpl parameters
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def get_final_predictions(self, pose_feat, shape_feat, cam_feat, bbox_info):
        # predict pose with CNN     
        pred_pose = self.pose_head(pose_feat, bbox_info) 
        pred_shape = self.shape_head(shape_feat).squeeze(1)
        pred_cam = self.cam_head(cam_feat.reshape(cam_feat.shape[0],1,-1)).squeeze(1)  # Linear Layer
        
        return pred_pose, pred_shape, pred_cam

    def forward(self, features, bbox_info=None):

        if isinstance(features, list) or isinstance(features, tuple):
            features = features[-1]

        batch_size = features.shape[0] # shape of [bs, c, h, w]
        
        init_pose = self.init_pose.expand(batch_size, -1).reshape(batch_size, self.num_joints, -1)   # N, Jx6
        init_shape = self.init_shape.expand(batch_size, -1) # [bs, 10]
        init_cam = self.init_cam.expand(batch_size, -1)     # [bs, 3]
        
        padding_mask = torch.zeros(batch_size, features.shape[-2], features.shape[-1]).to(features.device).bool() 
        pos = self.position_embedding(features, mask=padding_mask) # using None as mask will be a little faster
        #padding_mask = None
        features = self.input_proj(features)
        pose_feat, shape_feat, cam_feat, memory = self.transformer(
                                                features,
                                                padding_mask,
                                                self.pose_query.weight,
                                                self.shape_query.weight,
                                                self.cam_query.weight,
                                                pos)

        pred_pose, pred_shape, pred_cam = self.get_final_predictions(pose_feat, 
                                                               shape_feat, 
                                                               cam_feat,
                                                               bbox_info)
                                                               #init_pose=init_pose,
                                                               #init_shape=init_shape,
                                                               #init_cam=init_cam)
        
        pred_pose += init_pose
        pred_shape += init_shape
        pred_cam += init_cam
        
        pred_pose_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)
        output = {}
        output.update({
            'pred_pose': pred_pose_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
        })

        return output


#--------------------------------Position Encoding------------------------------------
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        # assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(position_encoding_type, hidden_dim):
    
    N_steps = hidden_dim // 2
    if position_encoding_type == "sine":
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_encoding_type == "learned":
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_encoding_type}")

    return position_embedding

