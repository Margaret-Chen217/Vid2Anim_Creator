import cv2
import torch
import os
import bpy
import pickle as pk
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

# import hybrik
from hybrik.models import builder as hybrik_builder
from hybrik.utils.vis import get_one_box
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLX

from . import util, config
from abc import ABC, abstractmethod

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

        #  save pt
        # outdir = "C:\\Users\\18523\\Desktop\\test\\denoise"
        # with open(os.path.join(outdir, 'data.pk'), 'wb') as fid:
        #     pk.dump(self.res_db, fid)

        # return self.res_db

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
