# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer for Human Mesh Reconstruction
Modified from original DETR implementation
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SimHMRFormer(nn.Module):

    def __init__(self,
                d_model=512, 
                nhead=8, 
                num_encoder_layers=6,
                num_decoder_layers=6, 
                dim_feedforward=2048, 
                dropout=0.1,
                activation="relu", 
                normalize_before=False,
                return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model,
                                                nhead, 
                                                dim_feedforward,
                                                dropout, 
                                                activation, 
                                                normalize_before)
        
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, 
                                          num_encoder_layers, 
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, 
                                                nhead, 
                                                dim_feedforward,
                                                dropout, 
                                                activation, 
                                                normalize_before)
        
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_decoder_layers, 
                                          decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src, 
                mask, 
                pose_query, 
                shape_query, 
                cam_query, 
                pos_embed):
        
        # B,C,H,W -> HW,B,C
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        pose_query = pose_query.unsqueeze(1).repeat(1, bs, 1)
        shape_query = shape_query.unsqueeze(1).repeat(1, bs, 1)
        cam_query = cam_query.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)
        #tgt = torch.zeros_like(pose_query)
        
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        pose_feat, shape_feat, cam_feat = self.decoder(memory, 
                                            memory_key_padding_mask=mask,
                                            pos=pos_embed, 
                                            pose_query=pose_query,
                                            shape_query=shape_query,
                                            cam_query=cam_query)
        # pose, shape, cam feat: [num_query, bs, dim]
        pose_feat = pose_feat.transpose(0,1) # [bs, 24, dim]
        shape_feat = shape_feat.transpose(0,1)
        cam_feat = cam_feat.transpose(0,1)

        return pose_feat, shape_feat, cam_feat, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = _get_clones(norm, 3) # individual norm layers for pose, shape, and camera
        self.return_intermediate = return_intermediate

    def forward(self,
                #tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pose_query: Optional[Tensor] = None,
                shape_query: Optional[Tensor] = None,
                cam_query: Optional[Tensor] = None):
        
        output_pose_embed = torch.zeros_like(pose_query)
        output_shape_embed = torch.zeros_like(shape_query)
        output_cam_embed = torch.zeros_like(cam_query)

        intermediate = []

        for layer in self.layers:
            output_pose_embed, output_shape_embed, output_cam_embed = layer(output_pose_embed,
                                                    output_shape_embed,
                                                    output_cam_embed,
                                                    memory,
                                                    tgt_mask=tgt_mask,
                                                    memory_mask=memory_mask,
                                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                                    memory_key_padding_mask=memory_key_padding_mask,
                                                    pos=pos,
                                                    pose_query=pose_query,
                                                    shape_query=shape_query,
                                                    cam_query=cam_query)
            
            if self.return_intermediate:
                intermediate.append([self.norm[0](output_pose_embed), self.norm[1](output_shape_embed), self.norm[2](output_cam_embed)])

        if self.norm is not None:
            output_pose_embed, output_shape_embed, output_cam_embed = self.norm[0](output_pose_embed),\
                                                                      self.norm[1](output_shape_embed),\
                                                                      self.norm[2](output_cam_embed)
        if self.return_intermediate:
            return intermediate
            # return torch.stack(intermediate)

        return output_pose_embed, output_shape_embed, output_cam_embed


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn_pose = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_pose = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_shape = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_cam = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.pose_linear1 = nn.Linear(d_model, dim_feedforward)
        self.pose_linear2 = nn.Linear(dim_feedforward, d_model)

        self.pose_norm1 = nn.LayerNorm(d_model)
        self.pose_norm2 = nn.LayerNorm(d_model)
        self.pose_norm3 = nn.LayerNorm(d_model)

        self.shape_linear1 = nn.Linear(d_model, dim_feedforward)
        self.shape_linear2 = nn.Linear(dim_feedforward, d_model)

        self.shape_norm1 = nn.LayerNorm(d_model)
        self.shape_norm2 = nn.LayerNorm(d_model)
        self.shape_norm3 = nn.LayerNorm(d_model)

        self.cam_linear1 = nn.Linear(d_model, dim_feedforward)
        self.cam_linear2 = nn.Linear(dim_feedforward, d_model)

        self.cam_norm1 = nn.LayerNorm(d_model)
        self.cam_norm2 = nn.LayerNorm(d_model)
        self.cam_norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     pose_tgt,
                     shape_tgt,
                     cam_tgt, 
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     pose_query: Optional[Tensor] = None,
                     shape_query: Optional[Tensor] = None,
                     cam_query: Optional[Tensor] = None):
        
        # self-attention on pose queries.
        q_pose = k_pose = self.with_pos_embed(pose_tgt, pose_query)
        pose_tgt2 = self.self_attn_pose(q_pose, k_pose, value=pose_tgt, attn_mask=tgt_mask,
                                   key_padding_mask=tgt_key_padding_mask)[0]
        
        pose_tgt = pose_tgt + self.dropout1(pose_tgt2)
        pose_tgt = self.pose_norm1(pose_tgt)

        # TODO: implement SA for shape and cam queries. 

        # cross-attention and FFN for pose queries.
        pose_tgt2 = self.cross_attn_pose(query=self.with_pos_embed(pose_tgt, pose_query),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        
        pose_tgt = pose_tgt + self.dropout2(pose_tgt2)
        pose_tgt = self.pose_norm2(pose_tgt)
        pose_tgt2 = self.pose_linear2(self.dropout(self.activation(self.pose_linear1(pose_tgt))))
        pose_tgt = pose_tgt + self.dropout3(pose_tgt2)
        pose_tgt = self.pose_norm3(pose_tgt)

        # cross-attention for shape and cam queries
        shape_tgt2 = self.cross_attn_shape(query=self.with_pos_embed(shape_tgt, shape_query),
                                              key=self.with_pos_embed(memory, pos),
                                              value=memory, attn_mask=memory_mask,
                                              key_padding_mask=memory_key_padding_mask)[0]
        
        cam_tgt2 = self.cross_attn_cam(query=self.with_pos_embed(cam_tgt, cam_query),
                                             key=self.with_pos_embed(memory, pos),
                                             value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask)[0]

        shape_tgt = shape_tgt + self.dropout2(shape_tgt2)
        shape_tgt = self.shape_norm2(shape_tgt)
        shape_tgt2 = self.shape_linear2(self.dropout(self.activation(self.shape_linear1(shape_tgt))))
        shape_tgt = shape_tgt + self.dropout3(shape_tgt2)
        shape_tgt = self.shape_norm3(shape_tgt)

        cam_tgt = cam_tgt + self.dropout2(cam_tgt2)
        cam_tgt = self.cam_norm2(cam_tgt)
        cam_tgt2 = self.cam_linear2(self.dropout(self.activation(self.cam_linear1(cam_tgt))))
        cam_tgt = cam_tgt + self.dropout3(cam_tgt2)
        cam_tgt = self.cam_norm3(cam_tgt)
        
        return pose_tgt, shape_tgt, cam_tgt

    # NOTE: This pre-norm version has not been modified for HMR
    def forward_pre(self,
                    tgt, 
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, 
                pose_tgt, 
                shape_tgt,
                cam_tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                pose_query: Optional[Tensor] = None,
                shape_query: Optional[Tensor] = None,
                cam_query: Optional[Tensor] = None
                ):
        if self.normalize_before:
            return self.forward_pre(pose_tgt,
                                    shape_tgt,
                                    cam_tgt, 
                                    memory, 
                                    tgt_mask, 
                                    memory_mask,
                                    tgt_key_padding_mask, 
                                    memory_key_padding_mask, 
                                    pos, 
                                    pose_query,
                                    shape_query,
                                    cam_query)
        
        return self.forward_post(pose_tgt,
                                 shape_tgt,
                                 cam_tgt, 
                                 memory, 
                                 tgt_mask, 
                                 memory_mask,
                                 tgt_key_padding_mask, 
                                 memory_key_padding_mask, 
                                 pos, 
                                 pose_query,
                                 shape_query,
                                 cam_query)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
