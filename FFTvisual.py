
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image

# 加载本地图像并转换为灰度
image = Image.open('data/VeRi/image_query/0030_c008_00049770_0.jpg').convert('L')

# 拉伸图像为256x256大小
image = image.resize((256, 256))
image_np = np.array(image)

# 进行快速傅里叶变换
f_transform = fft2(image_np)
f_transform_shifted = fftshift(f_transform)

# 构建频谱图
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

# 分离高频和低频成分
rows, cols = image_np.shape
crow, ccol = rows // 2 , cols // 2

# 创建掩膜，仅保留低频成分
low_freq_mask = np.zeros((rows, cols), dtype=int)
low_freq_radius = 30  # 调整这个值以控制低频保留的区域大小
low_freq_mask[crow-low_freq_radius:crow+low_freq_radius, ccol-low_freq_radius:ccol+low_freq_radius] = 1

# 创建掩膜，仅保留高频成分
high_freq_mask = 1 - low_freq_mask

# 应用掩膜
low_freq_component = f_transform_shifted * low_freq_mask
high_freq_component = f_transform_shifted * high_freq_mask

# 进行逆快速傅里叶变换
low_freq_image = np.abs(ifft2(fftshift(low_freq_component)))
high_freq_image = np.abs(ifft2(fftshift(high_freq_component)))

# 可视化原始图像
plt.figure()
plt.imshow(image_np, cmap='gray')
plt.axis('off')
plt.show()

# 可视化频谱图
plt.figure()
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')
plt.show()

# 可视化低频成分
plt.figure()
plt.imshow(low_freq_image, cmap='gray')
plt.axis('off')
plt.show()

# 可视化高频成分
plt.figure()
plt.imshow(high_freq_image, cmap='gray')
plt.axis('off')
plt.show()