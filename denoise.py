import numpy as np


def slidewindow_smooth(quaternions, joint_num, window_size=3):
    # 创建一个平均窗口
    window = np.ones(window_size) / window_size
    smoothed_quaternions = np.empty_like(quaternions)

    for i in range(joint_num):
        for j in range(4):  # 四个四元数分量分别平滑
            smoothed_quaternions[:, i, j] = np.convolve(
                quaternions[:, i, j], window, "same"
            )

    # 重新归一化四元数
    norms = np.linalg.norm(smoothed_quaternions, axis=2, keepdims=True)
    smoothed_quaternions /= norms

    return smoothed_quaternions


from scipy.signal import butter, filtfilt


def butterworth_smooth(matrix, joint_num, ):
    # 设定滤波器参数
    fs = 30  # 假设采样频率是30Hz
    cutoff = 5  # 截止频率3Hz
    order = 2  # 滤波器阶数

    # 设计滤波器
    b, a = butter(order, cutoff / (0.5 * fs), btype="low", analog=False)

    # 对每个四元数分量应用滤波器
    filtered_quaternions = np.empty_like(matrix)
    for i in range(joint_num):
        for j in range(4):  # 四元数的每个分量
            filtered_quaternions[:, i, j] = filtfilt(b, a, matrix[:, i, j])

    return filtered_quaternions

from scipy.fft import fft, ifft, fftfreq
def fourier_smooth(matrix, num_frames, num_joints,cutoff_ratio = 0.8):
    print(matrix.shape)
    quaternion_freqs = np.empty_like(matrix, dtype=np.complex)
    freqs = fftfreq(num_frames)

    for i in range(num_joints):
        for j in range(4):  # 对四元数的每个分量进行FFT
            quaternion_freqs[:, i, j] = fft(matrix[:, i, j])

    max_freq = np.abs(freqs).max()
    print("max_freq: ",max_freq)
    cutoff_frequency = cutoff_ratio * max_freq  # 设定截止频率为最大频率的20%
    print("cutoff: ", cutoff_frequency)
    # cutoff_frequency = 0.1  # 设定截止频率
    low_pass_filter = np.abs(freqs) < cutoff_frequency
    
    for i in range(num_joints):
        for j in range(4):
            quaternion_freqs[:, i, j] *= low_pass_filter 
    filtered_quaternions = np.empty_like(matrix)

    for i in range(num_joints):
        for j in range(4):
            filtered_quaternions[:, i, j] = ifft(quaternion_freqs[:, i, j]).real  # 取实部

    return filtered_quaternions

from scipy.signal import welch, wiener

def wiener_smooth(matrix,num_joints):
    filtered_quaternions = np.empty_like(matrix)
    for i in range(num_joints):
        for j in range(4):  # 四元数的每个分量
            # 边缘扩展
            # extended_data = np.pad(matrix[:, i, j], (1, 1), mode='edge')
            
            filtered_quaternions[:, i, j] = wiener(matrix[:, i, j], mysize=3, noise=0.01)
    return filtered_quaternions

def gaussian_smooth(matrix, num_joints,sigma = 0.5):
    kernel_size = 5  # 核大小
    kernel = np.exp(-np.square(np.arange(kernel_size) - kernel_size // 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # 归一化使总和为1
    smoothed_quaternions = np.empty_like(matrix)

    for i in range(num_joints):
        for j in range(4):  # 对四元数的每个分量进行卷积
            smoothed_quaternions[:, i, j] = np.convolve(matrix[:, i, j], kernel, mode='same')

    return smoothed_quaternions