from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
from torch import nn


def _ensure_bridgevla_imports():
    root = Path(__file__).resolve().parents[4]
    finetune_root = root / "BridgeVLA" / "finetune"
    point_renderer_root = finetune_root / "bridgevla" / "libs" / "point-renderer"
    for path in (finetune_root, point_renderer_root):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_ensure_bridgevla_imports()

import bridgevla.mvt.utils as mvt_utils  # noqa: E402
from bridgevla.mvt.mvt_single import MVT as MVTSingle  # noqa: E402


class BridgeFullMVT(nn.Module):
    """DeFi wrapper around BridgeVLA's original MVT core.

    This keeps the original point-renderer + PaliGemma MVT path, but strips
    out debug/visualization-only dependencies from BridgeVLA's `mvt.py`.
    """

    def __init__(
        self,
        depth: int,
        img_size: int,
        img_feat_dim: int,
        feat_dim: int,
        im_channels: int,
        activation: str,
        decoder_dropout: float,
        img_patch_size: int,
        final_dim: int,
        self_cross_ver: int,
        add_corr: bool,
        norm_corr: bool,
        add_pixel_loc: bool,
        add_depth: bool,
        rend_three_views: bool,
        use_point_renderer: bool,
        pe_fix: bool,
        feat_ver: int,
        wpt_img_aug: float,
        inp_pre_pro: bool,
        inp_pre_con: bool,
        cvx_up: bool,
        xops: bool,
        rot_ver: int,
        num_rot: int,
        stage_two: bool = False,
        st_sca: float = 4.0,
        st_wpt_loc_aug: float = 0.05,
        st_wpt_loc_inp_no_noise: bool = False,
        img_aug_2: float = 0.0,
        renderer_device: str = "cuda:0",
        load_pretrain: bool = False,
        pretrain_path: str | None = None,
    ):
        super().__init__()
        from point_renderer.rvt_renderer import RVTBoxRenderer as BoxRenderer

        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["BoxRenderer"]
        del args["stage_two"]
        del args["st_sca"]
        del args["st_wpt_loc_aug"]
        del args["st_wpt_loc_inp_no_noise"]
        del args["img_aug_2"]

        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2
        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=rend_three_views,
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.img_size = img_size
        self.mvt1 = MVTSingle(**args, renderer=self.renderer)

    def get_pt_loc_on_img(self, pt, mvt1_or_mvt2: bool, dyn_cam_info, out=None):
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3
        if mvt1_or_mvt2:
            assert out is None
            return self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)
        assert self.stage_two
        assert out is not None
        assert out["wpt_local1"].shape == (bs, 3)
        pt, _ = mvt_utils.trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
        pt = pt.view(bs, _np, 3)
        return self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)

    def get_wpt(self, out, mvt1_or_mvt2: bool, dyn_cam_info, y_q=None):
        if mvt1_or_mvt2:
            return self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        assert self.stage_two
        wpt = self.mvt1.get_wpt(out["mvt2"], dyn_cam_info, y_q)
        return out["rev_trans"](wpt)

    def render(self, pc, img_feat, img_aug, mvt1_or_mvt2: bool, dyn_cam_info):
        mvt = self.mvt1
        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info

            if mvt.add_corr:
                if mvt.norm_corr:
                    img = []
                    for _pc, _img_feat, _dyn_cam_info in zip(pc, img_feat, dyn_cam_info_itr):
                        max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                        img.append(
                            self.renderer(
                                _pc,
                                torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                fix_cam=True,
                                dyn_cam_info=(_dyn_cam_info,) if _dyn_cam_info is not None else None,
                            ).unsqueeze(0)
                        )
                else:
                    img = [
                        self.renderer(
                            _pc,
                            torch.cat((_pc, _img_feat), dim=-1),
                            fix_cam=True,
                            dyn_cam_info=(_dyn_cam_info,) if _dyn_cam_info is not None else None,
                        ).unsqueeze(0)
                        for (_pc, _img_feat, _dyn_cam_info) in zip(pc, img_feat, dyn_cam_info_itr)
                    ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,) if _dyn_cam_info is not None else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(pc, img_feat, dyn_cam_info_itr)
                ]

        img = torch.cat(img, 0).permute(0, 1, 4, 2, 3)
        if mvt.add_corr:
            mvt.img = img[:, :, 3:].clone().detach()
        else:
            mvt.img = img.clone().detach()
        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)
        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device)
            img = torch.cat((img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2)
        return img

    def forward(self, pc, img_feat, img_aug=0, wpt_local=None, rot_x_y=None, language_goal=None, **kwargs):
        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for x in img_feat:
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    x = x + noise
            img = self.render(pc=pc, img_feat=img_feat, img_aug=img_aug, mvt1_or_mvt2=True, dyn_cam_info=None)

        wpt_local_stage_one = wpt_local.clone().detach() if (self.training and wpt_local is not None) else wpt_local
        out = self.mvt1(
            img=img,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            language_goal=language_goal,
            forward_no_feat=False,
            **kwargs,
        )
        out["mvt1_ori_img"] = img.clone().detach()

        if self.stage_two:
            with torch.no_grad():
                if self.training:
                    wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                        wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
                    )
                    pc, rev_trans = mvt_utils.trans_pc(pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca)
                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = mvt_utils.trans_pc(wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca)
                    else:
                        wpt_local2, _ = mvt_utils.trans_pc(wpt_local, loc=wpt_local_stage_one, sca=self.st_sca)
                else:
                    wpt_local = self.get_wpt(out, y_q=None, mvt1_or_mvt2=True, dyn_cam_info=None)
                    pc, rev_trans = mvt_utils.trans_pc(pc, loc=wpt_local, sca=self.st_sca)
                    wpt_local_stage_one_noisy = wpt_local
                    wpt_local2 = None
                img = self.render(pc=pc, img_feat=img_feat, img_aug=img_aug, mvt1_or_mvt2=False, dyn_cam_info=None)

            out_mvt2 = self.mvt1(
                img=img,
                wpt_local=wpt_local2,
                rot_x_y=rot_x_y,
                language_goal=language_goal,
                forward_no_feat=False,
                **kwargs,
            )
            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans
            out["mvt2"] = out_mvt2
            out["mvt2_ori_img"] = img.clone().detach()

        return out

    def free_mem(self):
        self.renderer.free_mem()
