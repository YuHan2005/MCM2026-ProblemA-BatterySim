import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob
import warnings

# 忽略运行时的一些警告，保持输出整洁
warnings.filterwarnings('ignore')

# 绘图风格设置


class BatteryDatasetHandler:
    """
    专门处理NASA电池数据集（Patrick Fleith版本）的类
    """
    def __init__(self, dataset_path):
        """
        初始化数据集路径
        :param dataset_path: 包含 metadata.csv 和 data/ 文件夹的根目录
        """
        self.dataset_path = dataset_path
        self.metadata_file = os.path.join(dataset_path, 'metadata.csv')
        self.data_dir = os.path.join(dataset_path, 'data')
        
        # 加载元数据
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"未找到元数据文件: {self.metadata_file}")
        
        self.metadata = pd.read_csv(self.metadata_file)
        print(f"成功加载元数据，共 {len(self.metadata)} 条记录。")

    def get_battery_ids(self):
        """获取所有电池的ID列表"""
        return self.metadata['battery_id'].unique()

    def get_discharge_cycles(self, battery_id):
        """
        获取指定电池的所有放电循环元数据
        按 uid (操作顺序) 排序
        """
        battery_data = self.metadata[self.metadata['battery_id'] == battery_id]
        # 筛选 type 为 'discharge' 的记录
        discharge_data = battery_data[battery_data['type'] == 'discharge']
        return discharge_data.sort_values('uid')

    def load_cycle_timeseries(self, filename):
        """
        加载单个循环的详细CSV数据
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            # 尝试在子文件夹中查找（部分解压工具可能会创建嵌套结构）
            # 这里简化处理，假设在 data/ 根目录下
            return None
        return pd.read_csv(filepath)