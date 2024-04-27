import cv2
import torch
import os
import bpy
import pickle as pk
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from niki.utils.hybrik_utils.simple_transform_3d_smpl_cam import (
    SimpleTransform3DSMPLCam,
)
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

# import hybrik
from hybrik.models import builder as hybrik_builder
from niki.utils.hybrik_utils import builder as niki_builder
from hybrik.utils.vis import get_one_box
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX

from . import util, config
from abc import ABC, abstractmethod
from niki.models.NIKI_1stage import FlowIK_camnet
from niki.utils.demo_utils import *

# set gpu
gpu = 0


class Vid2Anim(ABC):
    def __init__(self, context) -> None:

        self.context = context
        self.use_gpu = self.context.scene.use_gpu

    @abstractmethod
    def prepare():
        pass

    @abstractmethod
    def run():
        pass


class Hybrik(Vid2Anim):
    def __init__(self, context) -> None:
        # set config
        self.det_transform = T.Compose([T.ToTensor()])

        self.cfg_file = "./configs/smplx/256x192_hrnet_rle_smplx_kid.yaml"
        self.CKPT = "./pretrained_models/hybrikx_rle_hrnet.pth"
        self.cfg = update_config(self.cfg_file)
        self.cfg["MODEL"]["EXTRA"]["USE_KID"] = self.cfg["DATASET"].get(
            "USE_KID", False
        )
        self.cfg["LOSS"]["ELEMENTS"]["USE_KID"] = self.cfg["DATASET"].get(
            "USE_KID", False
        )
        # 获取模型配置中BBOX_3D_SHAPE字段值,如果没有就默认为(2000,2000,2000)
        self.bbox_3d_shape = getattr(
            self.cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000)
        )
        # 将bbox_3d_shape每个值乘以1e-3,转换为米为单位
        self.bbox_3d_shape = [item * 1e-3 for item in self.bbox_3d_shape]

        # 创建一个字典dummy_set,保存关键点对信息和bbox_3d_shape
        self.dummpy_set = edict(
            {
                "joint_pairs_17": None,
                "joint_pairs_24": None,
                "joint_pairs_29": None,
                "bbox_3d_shape": self.bbox_3d_shape,
            }
        )

        self.res_db = {k: [] for k in config.res_keys_hybrik}

        # 初始化图像和3D姿态转换类,设置各种参数
        self.transformation = SimpleTransform3DSMPLX(
            self.dummpy_set,
            scale_factor=self.cfg.DATASET.SCALE_FACTOR,
            color_factor=self.cfg.DATASET.COLOR_FACTOR,
            occlusion=self.cfg.DATASET.OCCLUSION,
            input_size=self.cfg.MODEL.IMAGE_SIZE,
            output_size=self.cfg.MODEL.HEATMAP_SIZE,
            depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self.cfg.DATASET.ROT_FACTOR,
            sigma=self.cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=self.cfg.LOSS["TYPE"],
        )

        super().__init__(context)

    def prepare(self):
        self.det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.hybrik_model = hybrik_builder.build_sppe(self.cfg.MODEL)
        print(f"Loading model from {self.CKPT}...")

        # 加载预训练权重参数
        save_dict = torch.load(self.CKPT, map_location="cpu")

        if type(save_dict) == dict:
            # 获取模型权重字典, 加载Hybrik模型权重
            model_dict = save_dict["model"]
            self.hybrik_model.load_state_dict(model_dict)
        else:
            # 加载Hybrik模型权重
            self.hybrik_model.load_state_dict(save_dict)

        if self.use_gpu:
            # 将检测模型放入GPU
            self.det_model.cuda(gpu)
            # 将Hybrik模型放入GPU
            self.hybrik_model.cuda(gpu)
        # 设置检测模型为评估模式
        self.det_model.eval()
        # 设置Hybrik模型为评估模式
        self.hybrik_model.eval()

    def run(self, img_path_list, denoise=None):
        print("use_gpu: ", self.use_gpu)
        print("use_denoise: ", denoise)
        for k in self.res_db.keys():
            print(k)

        # 预测
        # 初始化上一帧检测框
        prev_box = None
        print("### Run Model...")
        for img_path in tqdm(img_path_list):
            with torch.no_grad():
                # Run Detection

                # 读取图像
                input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

                # 从检测结果中获取当前帧的人体框架区域坐标:
                # 预处理输入检测模型
                if self.use_gpu:
                    det_input = self.det_transform(input_image).to(gpu)
                else:
                    det_input = self.det_transform(input_image)
                # 运行检测模型预测框
                det_output = self.det_model([det_input])[0]

                # 如果当前帧是第一帧(prev_box是None)
                # 直接取检测结果的最大置信度框作为tight_bbox
                if prev_box is None:
                    tight_bbox = get_one_box(det_output)  # xyxy
                    if tight_bbox is None:
                        continue
                else:
                    # 首先取检测结果的最大置信度框作为tight_bbox
                    tight_bbox = get_one_box(det_output)  # xyxy
                # 如果上面都没有获取到框,则使用上一帧框
                if tight_bbox is None:
                    tight_bbox = prev_box
                # 最后更新prev_box为当前帧的tight_bbox
                prev_box = tight_bbox

                # Run HybrIK
                # bbox: [x1, y1, x2, y2]
                # 使用变换将图像和bbox从图像坐标转换到模型坐标
                pose_input, bbox, img_center = self.transformation.test_transform(
                    input_image.copy(), tight_bbox
                )
                # 将输入放入GPU

                if self.use_gpu:
                    pose_input = pose_input.to(gpu)[None, :, :, :]
                else:
                    pose_input = pose_input[None, :, :, :]

                # vis 2d
                # 将bbox格式转换为xywh
                bbox_xywh = util.xyxy2xywh(bbox)

                # 使用Hybrik模型进行预测
                pose_output = self.hybrik_model(
                    pose_input,
                    flip_test=True,
                    bboxes=torch.from_numpy(np.array(bbox))
                    .to(pose_input.device)
                    .unsqueeze(0)
                    .float(),
                    img_center=torch.from_numpy(img_center)
                    .to(pose_input.device)
                    .unsqueeze(0)
                    .float(),
                )
                # 提取3D位移预测
                transl = pose_output.transl.detach()

                # 设置焦距
                focal = 1000.0
                bbox_xywh = util.xyxy2xywh(bbox)
                # 转换位移坐标系
                transl_camsys = transl.clone()
                transl_camsys = transl_camsys * 256 / bbox_xywh[2]

                # 根据bbox大小转换焦距
                focal = focal / 256 * bbox_xywh[2]
                # 获取关键点预测
                vertices = pose_output.pred_vertices.detach()
                # 将数据格式转换为渲染需要的格式

            assert (
                pose_input.shape[0] == 1
            ), "Only support single batch inference for now"
            # 提取2D关键点预测结果
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(-1, 3).cpu().data.numpy()
            # 提取关键点置信度预测
            pred_scores = pose_output.maxvals.cpu().data[:, :29].reshape(29).numpy()
            # 提取相机参数预测
            pred_camera = pose_output.pred_camera.squeeze(dim=0).cpu().data.numpy()
            # 提取形状参数预测
            pred_betas = pose_output.pred_shape_full.squeeze(dim=0).cpu().data.numpy()
            # 提取姿态矩阵预测
            pred_theta = pose_output.pred_theta_mat.squeeze(dim=0).cpu().data.numpy()
            # 提取姿态矩阵预测
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            # 提取全局旋转预测
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            # 输入图像大小

            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            # 将每帧结果添加到对应字段列表
            self.res_db["pred_uvd"].append(pred_uvd_jts)
            self.res_db["pred_scores"].append(pred_scores)
            self.res_db["pred_camera"].append(pred_camera)
            self.res_db["f"].append(1000.0)
            self.res_db["pred_betas"].append(pred_betas)
            self.res_db["pred_thetas"].append(pred_theta)
            self.res_db["pred_phi"].append(pred_phi)
            self.res_db["pred_cam_root"].append(pred_cam_root)
            self.res_db["transl"].append(transl[0].cpu().data.numpy())
            self.res_db["transl_camsys"].append(transl_camsys[0].cpu().data.numpy())
            self.res_db["bbox"].append(np.array(bbox))
            self.res_db["height"].append(img_size[0])
            self.res_db["width"].append(img_size[1])
            self.res_db["img_path"].append(img_path)

        self.res_db["joint_num"] = 55
        n_frames = len(self.res_db["img_path"])
        for k in self.res_db.keys():
            if k == "joint_num":
                continue
            mat = np.array(self.res_db[k])
            print(k, ": ", mat.shape)
            self.res_db[k] = np.stack(self.res_db[k])
            assert self.res_db[k].shape[0] == n_frames

        # denoise
        if denoise == None:
            return self.res_db
        else:
            mat_copy = self.res_db["pred_thetas"]
            if denoise == "butterworth":
                if bpy.app.debug:
                    print("Denoise Mode: butterworth")
                mat_copy = self.res_db["pred_thetas"]
                print(mat_copy.shape)
                self.res_db["pred_thetas"] = util.butterworth_denoise(
                    mat_copy, joint_num=55
                )
                return self.res_db

            if denoise == "conv_avg":
                if bpy.app.debug:
                    print("Denoise Mode: conv_avg")
                self.res_db["pred_thetas"] = util.conv_avg_denoise(
                    mat_copy, joint_num=55
                )

            if denoise == "conv_gaussian":
                if bpy.app.debug:
                    print("Denoise Mode: conv_gaussian")
                pass

            return self.res_db

class Niki(Vid2Anim):
    def __init__(self, context) -> None:
        self.det_transform = T.Compose([T.ToTensor()])

        self.cfg_file = "configs/hybrik_config.yaml"
        self.CKPT = "pretrained_models/checkpoint_49_cocoeft.pth"
        self.cfg = update_config(self.cfg_file)

        self.v_cfg_file = "configs/NIKI-1stage.yaml"
        self.V_CKPT = "pretrained_models/niki_model_28.pth"
        self.v_cfg = update_config(self.v_cfg_file)

        # 获取模型配置中BBOX_3D_SHAPE字段值,如果没有就默认为(2000,2000,2000)
        self.bbox_3d_shape = getattr(
            self.cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000)
        )
        # 将bbox_3d_shape每个值乘以1e-3,转换为米为单位
        self.bbox_3d_shape = [item * 1e-3 for item in self.bbox_3d_shape]

        # 创建一个字典dummy_set,保存关键点对信息和bbox_3d_shape
        self.dummpy_set = edict(
            {
                "joint_pairs_17": None,
                "joint_pairs_24": None,
                "joint_pairs_29": None,
                "bbox_3d_shape": self.bbox_3d_shape,
            }
        )
        self.res_db = {k: [] for k in config.res_keys_niki}

        self.transformation = SimpleTransform3DSMPLCam(
            self.dummpy_set,
            scale_factor=self.cfg.DATASET.SCALE_FACTOR,
            color_factor=self.cfg.DATASET.COLOR_FACTOR,
            occlusion=self.cfg.DATASET.OCCLUSION,
            input_size=self.cfg.MODEL.IMAGE_SIZE,
            output_size=self.cfg.MODEL.HEATMAP_SIZE,
            depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self.cfg.DATASET.ROT_FACTOR,
            sigma=self.cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=self.cfg.LOSS["TYPE"],
        )

        super().__init__(context)

    def prepare(self):
        self.det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.hybrik_model = niki_builder.build_sppe(self.cfg.MODEL)
        self.flow_model = FlowIK_camnet(self.v_cfg)

        print(f"Loading model from {self.CKPT}...")
        save_dict = torch.load(self.CKPT, map_location="cpu")
        if type(save_dict) == dict:
            model_dict = save_dict["model"]
            self.hybrik_model.load_state_dict(model_dict)
        else:
            self.hybrik_model.load_state_dict(save_dict)

        print(f"Loding LGD model from {self.V_CKPT}")
        save_dict = torch.load(self.V_CKPT, map_location="cpu")
        self.flow_model.load_state_dict(save_dict, strict=False)

        camnet_dict = "pretrained_models/niki_model_28.pth"
        tmp_dict = torch.load(camnet_dict)
        new_tmp_dict = {}
        for k, v in tmp_dict.items():
            if "regressor.camnet" in k:
                new_k = k[len("regressor.camnet.") :]
                new_tmp_dict[new_k] = v

        self.flow_model.regressor.camnet.load_state_dict(new_tmp_dict)

        self.det_model.cuda(gpu)
        self.hybrik_model.cuda(gpu)
        self.flow_model.cuda(gpu)
        self.det_model.eval()
        self.hybrik_model.eval()
        self.flow_model.eval()

    def run(self, img_path_list, denoise=False):
        prev_box = None
        print("### Run Model...")
        for img_path in tqdm(img_path_list, dynamic_ncols=True):

            with torch.no_grad():
                # Run Detection
                input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                det_input = self.det_transform(input_image).to(gpu)
                det_output = self.det_model([det_input])[0]

                if prev_box is None:
                    tight_bbox = get_one_box(det_output)  # xyxy
                    if tight_bbox is None:
                        continue
                else:
                    tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

                    area = (tight_bbox[2] - tight_bbox[0]) * (
                        tight_bbox[3] - tight_bbox[1]
                    )

                    max_bbox = get_one_box(det_output)  # xyxy
                    if max_bbox is not None:
                        max_area = (max_bbox[2] - max_bbox[0]) * (
                            max_bbox[3] - max_bbox[1]
                        )
                        if area < max_area * 0.1:
                            tight_bbox = max_bbox

                prev_box = tight_bbox

                # Run HybrIK
                # bbox: [x1, y1, x2, y2]
                pose_input, bbox, img_center = self.transformation.test_transform(
                    input_image, tight_bbox
                )
                pose_input = pose_input.to(gpu)[None, :, :, :]
                pose_output = self.hybrik_model(
                    pose_input,
                    flip_test=False,
                    bboxes=torch.from_numpy(np.array(bbox))
                    .to(pose_input.device)
                    .unsqueeze(0)
                    .float(),
                    img_center=torch.from_numpy(img_center)
                    .to(pose_input.device)
                    .unsqueeze(0)
                    .float(),
                    do_hybrik=False,
                )

                # === Save PT ===
                assert (
                    pose_input.shape[0] == 1
                ), "Only support single batch inference for now"

                pred_uvd_jts = (
                    pose_output.pred_uvd_jts.reshape(-1, 3).cpu().data.numpy()
                )
                pred_xyz_jts_29 = (
                    pose_output.pred_xyz_jts_29.reshape(-1, 3).cpu().data.numpy()
                )
                pred_scores = pose_output.maxvals.cpu().data[:, :29].reshape(29).numpy()
                pred_betas = pose_output.pred_shape.squeeze(dim=0).cpu().data.numpy()
                pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
                pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
                pred_sigma = pose_output.sigma.cpu().data.numpy()

                img_size = np.array((input_image.shape[1], input_image.shape[0]))

                self.res_db["pred_uvd"].append(pred_uvd_jts)
                self.res_db["pred_xyz_29"].append(pred_xyz_jts_29)
                self.res_db["pred_scores"].append(pred_scores)
                self.res_db["pred_sigma"].append(pred_sigma)
                self.res_db["f"].append(1000.0)
                self.res_db["pred_betas"].append(pred_betas)
                self.res_db["pred_phi"].append(pred_phi)
                self.res_db["pred_cam_root"].append(pred_cam_root)
                self.res_db["bbox"].append(np.array(bbox))
                self.res_db["height"].append(img_size[1])
                self.res_db["width"].append(img_size[0])
                self.res_db["img_path"].append(img_path)
                self.res_db["img_sizes"].append(img_size)

        total_img = len(self.res_db["img_path"])

        for k in self.res_db.keys():
            mat = np.array(self.res_db[k])
            print(k, ": ", mat.shape)
            try:
                v = np.stack(self.res_db[k], axis=0)
            except Exception:
                v = self.res_db[k]
                print(k, " failed")

            self.res_db[k] = v

        # FORWARD
        seq_len = 16
        video_res_db = {}
        total_img = (total_img // seq_len) * seq_len
        video_res_db["transl"] = torch.zeros((total_img, 3))
        video_res_db["vertices"] = torch.zeros((total_img, 6890, 3))
        video_res_db["img_path"] = self.res_db["img_path"]
        video_res_db["bbox"] = torch.zeros((total_img, 4))
        video_res_db["pred_uv"] = torch.zeros((total_img, 29, 2))
        video_res_db["pred_thetas"] = torch.zeros((total_img, 24, 3, 3))

        mean_beta = self.res_db["pred_betas"].mean(axis=0)
        self.res_db["pred_betas"][:] = mean_beta

        update_bbox = self.v_cfg.get("update_bbox", False)
        USE_HYBRIK_CAM = True

        idx = 0
        for i in tqdm(range(0, total_img - seq_len + 1, seq_len), dynamic_ncols=True):
            pred_xyz_29 = self.res_db["pred_xyz_29"][i : i + seq_len, :, :] * 2.2
            pred_uv = self.res_db["pred_uvd"][i : i + seq_len, :, :2]
            pred_sigma = self.res_db["pred_sigma"][i : i + seq_len, :, :].squeeze(1)
            pred_beta = self.res_db["pred_betas"][i : i + seq_len, :]
            pred_phi = self.res_db["pred_phi"][i : i + seq_len, :]
            pred_cam_root = self.res_db["pred_cam_root"][i : i + seq_len, :]
            pred_cam = np.concatenate(
                (1000.0 / (256 * pred_cam_root[:, [2]] + 1e-9), pred_cam_root[:, :2]),
                axis=1,
            )
            bbox = self.res_db["bbox"][i : i + seq_len, :]  # xyxy

            pred_xyz_29 = pred_xyz_29 - pred_xyz_29[:, [1, 2], :].mean(
                axis=1, keepdims=True
            )

            bbox_cs = xyxy_to_center_scale_batch(bbox)
            inp = {
                "pred_xyz_29": pred_xyz_29,
                "pred_uv": pred_uv,
                "pred_sigma": pred_sigma,
                "pred_beta": pred_beta,
                "pred_phi": pred_phi,
                "pred_cam": pred_cam,
                "bbox": bbox_cs,
                "img_sizes": self.res_db["img_sizes"][i : i + seq_len, :],
            }

            for k in inp.keys():
                inp[k] = torch.from_numpy(inp[k]).float().cuda().unsqueeze(0)

            if update_bbox:
                inp = reproject_uv(inp)
            else:
                img_center = (
                    (inp["img_sizes"] * 0.5 - inp["bbox"][:, :, :2])
                    / inp["bbox"][:, :, 2:]
                    * 256.0
                )
                inp["img_center"] = img_center

            with torch.no_grad():
                output = self.flow_model.forward_getcam(inp=inp)

                video_res_db["vertices"][i : i + seq_len] = output.verts.cpu()[0]
                video_res_db["bbox"][i : i + seq_len] = inp["bbox"][0].cpu()
                video_res_db["transl"][i : i + seq_len] = output.transl.cpu()[0]
                video_res_db["pred_uv"][i : i + seq_len] = output.inv_pred2uv.cpu()[0]
                video_res_db["pred_thetas"][i : i + seq_len] = output.rotmat.cpu()[0]
                if USE_HYBRIK_CAM:
                    video_res_db["transl"][i : i + seq_len] = torch.from_numpy(
                        self.res_db["pred_cam_root"][i : i + seq_len]
                    )

            n_frames = len(self.res_db["img_path"])
            video_res_db["pred_thetas"] = np.stack(video_res_db["pred_thetas"])
            video_res_db["joint_num"] = 25
            video_res_db["pred_cam_root"] = self.res_db["pred_cam_root"]
            # with open(os.path.join(opt.out_dir, 'res_01.pk'), 'wb') as fid:
            #     pk.dump(video_res_db, fid)

            # denoise
            if denoise:
                mat_copy = video_res_db["pred_thetas"]
                print(mat_copy.shape)
                video_res_db["pred_thetas"] = util.butterworth_denoise(
                    mat_copy, joint_num=24
                )

            return video_res_db
