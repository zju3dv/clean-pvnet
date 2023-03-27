"""
datasets:数据集加载库
=====================

用于加载本项目需要的各种数据集(LINEMOD, T-Less, custom)。
基于 ``pytorch.utils.data`` 库内的基类构建本框架，分为三个部分：

1. 数据集预处理(torch.utils.data.Dataset):

   - 初步封装数据集，实现按索引访问数据集和获取数据集长度的功能
   - 实现数据增强的功能：图像随机几何空间变换、随机颜色空间变换、随机像素空间变换(滤波)
   - 对数据进行了归一化和标准化

2. 数据集采样(torch.utils.dataSampler):
   
   a. 单样本采样:在Dataset的基础之上,返回数据集索引的可迭代对象(iterable)，以便对数据集进行遍历.
      分为两种具体的实现方法：

      - ``RandomSampler`` :乱序返回可迭代的索引对象
      - ``SequentialSampler`` :顺序返回可迭代的索引对象

   b. 批量采样：对单采样样本进一步封装，同样返回一个可迭代对象，但每次迭代返回的索引值的集合，以便实现批量访问数据.
      同样分为两种具体的实现方法:

      - ``IterationBasedBatchSampler`` :可循环对数据集索引迭代直至达到指定的次数
      - ``ImageSizeBatchSampler`` :每次迭代时不仅返回一个索引值的集合,而且还返回了一个随机生成的图像尺寸(用于图像的几何变换)

3. 数据集加载(torch.utils.data.Dataloader):结合 ``Sampler`` 和 ``Dataset`` 创建了一个 ``DataLoader`` 实例.
   本项目后续所有对数据集的操作都基于该实例进行,也是本库最终产生的对外接口.


.. note:: 关于本自建库的几点注意事项:
          
          1. 在DataLoader多进程的使用中注意子进程numpy随机数种子的初始化问题,具体可见 ``make_dataset.make_data_loader``
          2. 在具体实现代码中注意PIL与openv(numpy)处理图像的方式不同：

             - PIL的图像保存格式为RGB, 宽*高(当其转换为numpy格式后会自动变为高*宽，但通道顺序未变)
             - opencv(numpy)的图像格式为BFR,高*宽  
"""
from .make_dataset import make_data_loader