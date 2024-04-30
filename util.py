import os
import bpy
import cv2
import torch
import numpy as np
from scipy.signal import butter, filtfilt, convolve2d


def video2imageSeq(context, tempfolder_path):
    """_summary_

    Args:
        context (_type_): _description_
        tempfolder_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    video_path = context.scene.sourcefile_path
    video_basename = os.path.basename(video_path)
    rawimages_path = os.path.join(tempfolder_path, "raw_images")

    # _, info, _ = hybrik_util.get_video_info(video_path)

    if not os.path.exists(rawimages_path):
        os.makedirs(rawimages_path)
    os.system(f"ffmpeg -i {video_path} {rawimages_path}/{video_basename}-%06d.png")
    files = os.listdir(rawimages_path)
    files.sort()
    img_path_list = []

    wm = bpy.context.window_manager
    bpy.context.window_manager.progress_begin(0, len(files))

    for index, file in enumerate(files):
        if not os.path.isdir(file) and file[-4:] in [".jpg", ".png"]:
            img_path = os.path.join(rawimages_path, file)
            img_path_list.append(img_path)
        wm.progress_update(index)
    wm.progress_end
    return img_path_list


def butterworth_denoise(
    mat_origin, order=4, cutoff_freq=0.1, sampling_freq=1, joint_num=55
):
    # 计算滤波器的分子和分母系数
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff_freq, btype="low", analog=False)

    print("matrix shape: ", mat_origin.shape)
    frame_num = mat_origin.shape[0]

    matrix = []

    for f in range(frame_num):
        pose = mat_origin[f]
        frame_matrix = np.asarray(pose).reshape(joint_num, 3, 3)
        row = []
        for j in range(joint_num):
            row.append(frame_matrix[j])

        matrix.append(row)

    matrix = np.array(matrix)
    print(matrix.shape)
    # print(matrix)

    # 创建一个新的数组存储滤波后的矩阵

    filtered_matrix = np.zeros_like(matrix)

    for j in range(joint_num):
        for i in range(3):
            for k in range(3):
                filtered_matrix[:, j, i, k] = filtfilt(b, a, matrix[:, j, i, k])
    print(filtered_matrix.shape)
    # print(filtered_matrix)

    filtered_pred_thetas = filtered_matrix.reshape(frame_num, -1)
    print(filtered_pred_thetas.shape)
    # print(filtered_pred_thetas)

    return filtered_pred_thetas


def conv_avg_denoise(mat_origin, kernel_size=3, joint_num=55):
    print("matrix shape: ", mat_origin.shape)
    frame_num = mat_origin.shape[0]

    matrix = []

    for f in range(frame_num):
        pose = mat_origin[f]
        frame_matrix = np.asarray(pose).reshape(joint_num, 3, 3)
        matrix.append(frame_matrix)

    matrix = np.array(matrix)
    print(matrix.shape)

    # 创建一个新的数组存储卷积平均后的矩阵
    filtered_matrix = np.zeros_like(matrix)

    # 创建二维卷积核
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

    # 对每个关节的旋转矩阵进行卷积操作
    for j in range(joint_num):
        for i in range(3):
            filtered_matrix[:, j, i, :] = convolve2d(
                matrix[:, j, i, :], kernel, mode="same"
            )

    print(filtered_matrix.shape)

    filtered_pred_thetas = filtered_matrix.reshape(frame_num, -1)
    print(filtered_pred_thetas.shape)

    return filtered_pred_thetas


def recognize_video_ext(ext=""):
    if ext == "mp4":
        return cv2.VideoWriter_fourcc(*"mp4v"), "." + ext
    elif ext == "avi":
        return cv2.VideoWriter_fourcc(*"XVID"), "." + ext
    elif ext == "mov":
        return cv2.VideoWriter_fourcc(*"XVID"), "." + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"


def integral_hm(hms):
    # hms: [B, K, H, W]
    B, K, H, W = hms.shape
    hms = hms.sigmoid()
    hms = hms.reshape(B, K, -1)
    hms = hms / hms.sum(dim=2, keepdim=True)
    hms = hms.reshape(B, K, H, W)

    hm_x = hms.sum((2,))
    hm_y = hms.sum((3,))

    w_x = torch.arange(hms.shape[3]).to(hms.device).float()
    w_y = torch.arange(hms.shape[2]).to(hms.device).float()

    hm_x = hm_x * w_x
    hm_y = hm_y * w_y

    coord_x = hm_x.sum(dim=2, keepdim=True)
    coord_y = hm_y.sum(dim=2, keepdim=True)

    coord_x = coord_x / float(hms.shape[3]) - 0.5
    coord_y = coord_y / float(hms.shape[2]) - 0.5

    coord_uv = torch.cat((coord_x, coord_y), dim=2)
    return coord_uv


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), "Cannot capture source"
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (
        int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {"fourcc": fourcc, "fps": fps, "frameSize": frameSize}
    stream.release()

    return stream, videoinfo, datalen


import os
import pickle as pk

import bpy
import numpy as np

from config import x_part_match, part_match


def rot2quat(rot):
    """将旋转矩阵转换成四元数

    Args:
        rot (_type_): _description_

    Returns:
        _type_: _description_
    """
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot.reshape(9)
    q_abs = np.array(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ]
    )
    q_abs = np.sqrt(np.maximum(q_abs, 0))

    quat_by_rijk = np.vstack(
        [
            np.array([q_abs[0] ** 2, m21 - m12, m02 - m20, m10 - m01]),
            np.array([m21 - m12, q_abs[1] ** 2, m10 + m01, m02 + m20]),
            np.array([m02 - m20, m10 + m01, q_abs[2] ** 2, m12 + m21]),
            np.array([m10 - m01, m20 + m02, m21 + m12, q_abs[3] ** 2]),
        ]
    )
    flr = 0.1
    quat_candidates = quat_by_rijk / np.maximum(2.0 * q_abs[:, None], 0.1)

    idx = q_abs.argmax(axis=-1)

    quat = quat_candidates[idx]
    return quat


def deg2rad(angle):
    """将角度从度数转换为弧度

    Args:
        angle (_type_): _description_

    Returns:
        _type_: _description_
    """
    return -np.pi * (angle + 90) / 180.0


def init_scene(self, root_path, joint_num):
    # load fbx model
    print("joint_num init = ", joint_num)
    if joint_num == 55:
        bpy.ops.import_scene.fbx(
            filepath=os.path.join(root_path, "data", "smplx-neutral.fbx"),
            axis_forward="-Y",
            axis_up="-Z",
            global_scale=1,
        )
        obname = "SMPLX-mesh-neutral"
        arm_obname = "SMPLX-neutral"
    else:
        gender = "m"
        bpy.ops.import_scene.fbx(
            filepath=os.path.join(
                root_path, "data", f"basicModel_{gender}_lbs_10_207_0_v1.0.2.fbx"
            ),
            axis_forward="-Y",
            axis_up="-Z",
            global_scale=100,
        )
        obname = "%s_avg" % gender[0]
        arm_obname = "Armature"

    print("success load fbx")
    self.report({"INFO"}, "Load FBX " + obname)
    ob = bpy.data.objects[obname]

    ob.active_material = bpy.data.materials["Material"]

    cam_ob = bpy.data.objects["Camera"]
    cam_ob.location = [0, 0, 0]
    cam_ob.rotation_euler = [np.pi / 2, 0, 0]

    arm_ob = bpy.data.objects[arm_obname]
    arm_ob.animation_data_clear()

    return (ob, obname, arm_ob)


def Rodrigues(rotvec):
    """将旋转向量转换为旋转矩阵

    Args:
        rotvec (_type_): _description_

    Returns:
        _type_: _description_
    """
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


def rotate180(rot):
    """绕y轴和z轴旋转180度

    Args:
        rot (_type_): _description_

    Returns:
        _type_: _description_
    """
    xyz_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    return np.dot(xyz_convert.T, rot)


def convert_transl(transl):
    """从右手坐标系转换到左手坐标系
    y轴z轴取反

    Args:
        transl (_type_): _description_

    Returns:
        _type_: _description_
    """

    xyz_convert = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)

    if type(transl) == torch.Tensor:
        return transl.numpy().dot(xyz_convert)

    else:
        # type(transl) == np.ndarray
        return transl.dot(xyz_convert)


def rodrigues2bshapes(pose):
    """旋转向量转换为旋转矩阵和形状参数

    Args:
        pose (_type_): _description_

    Returns:
        _type_: _description_
    """
    if pose.size == 24 * 9:
        rod_rots = np.asarray(pose).reshape(24, 3, 3)
        mat_rots = [rod_rot for rod_rot in rod_rots]
    elif pose.size == 55 * 9:
        rod_rots = np.asarray(pose).reshape(55, 3, 3)
        mat_rots = [rod_rot for rod_rot in rod_rots]
    else:
        rod_rots = np.asarray(pose).reshape(24, 3)
        mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate(
        [(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]]
    )
    return (mat_rots, bshapes)


def setState0():
    """
    取消选择所有物体
    """
    for ob in bpy.data.objects.values():
        ob.select = False
    bpy.context.scene.objects.active = None


# apply trans pose and shape to character
def apply_trans_pose_shape(trans, pose, shape, ob, arm_ob, obname, scene, frame=None):

    # 转换坐标系F
    trans = convert_transl(trans)

    # 匹配骨骼dict

    selected_part_match = x_part_match

    # set the location of the first bone to the translation parameter
    # arm_ob.pose.bones[obname + '_Pelvis'].location = trans
    # 应用到root
    arm_ob.pose.bones["root"].location = trans
    arm_ob.pose.bones["root"].keyframe_insert("location", frame=frame)
    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(pose):
        bone = arm_ob.pose.bones[selected_part_match["bone_%02d" % ibone]]
        bone.rotation_quaternion = mrot
        if frame is not None:
            bone.keyframe_insert("rotation_quaternion", frame=frame)
            bone.keyframe_insert("location", frame=frame)


def load_bvh(self, res_db, root_path):
    scene = bpy.data.scenes["Scene"]

    joint_num = 55
    ob, obname, arm_ob = init_scene(self, root_path, joint_num)

    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    # clear all animation data
    arm_ob.animation_data_clear()
    # cam_ob.animation_data_clear()
    # load smpl params:
    nFrames = len(res_db["pred_thetas"])

    bpy.context.scene.frame_end = nFrames

    all_betas = res_db["pred_betas"]
    avg_beta = np.mean(all_betas, axis=0)

    # 对每一帧
    # 对第一个旋转矩阵mrots[0]进行180度旋转,使其与角色的初始姿态对齐

    mat = res_db["pred_thetas"]

    mat = mat.reshape((nFrames, joint_num, 3, 3))
    print(mat.shape)
    for frame in range(nFrames):
        mat[frame][0] = rotate180(mat[frame][0])

    # 转为四元数
    mat_qua = np.zeros((nFrames, joint_num, 4))

    for frame in range(nFrames):
        pose = mat[frame]
        for ibone, mrot in enumerate(pose):
            quaternion = rot2quat(mrot)
            mat_qua[frame, ibone] = quaternion

    # denoise

    for frame in range(nFrames):
        print(frame)
        scene.frame_set(frame)

        trans = res_db["transl_camsys"][frame]
        shape = avg_beta
        pose = mat_qua[frame]

        apply_trans_pose_shape(
            trans, pose, shape, ob, arm_ob, obname, scene, frame=frame
        )
        # scene.update()
