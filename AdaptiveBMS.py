import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from BatteryV3 import PhysicsBattery

# ==========================================
# 1. 修正后的 BMS 类 (参数与逻辑完全对齐)
# ==========================================
class AdaptiveBMS:
    def __init__(self, battery_model, learning_rate_R=5e-5, feedback_gain_soc=0.05):
        """
        修正点：默认 learning_rate_R 提高到 5e-5，与你手动运行的参数一致
        """
        self.battery = battery_model
        self.lr_r = learning_rate_R
        self.gain_soc = feedback_gain_soc
        
        self.logs = {'r_history': [], 'soc_correction': []}

    def update(self, v_meas, v_est, current, dt, step_i):
        """实时修正逻辑 (保持你满意的逻辑不变)"""
        voltage_error = v_meas - v_est
        soc_adjust = 0.0

        # --- A. SOC 修正 ---
        if abs(current) > 0.1 and step_i > 10:
            soc_correction = self.gain_soc * voltage_error * dt
            soc_adjust = soc_correction
            
            # 计算电量变化 (Ah)
            delta_q = soc_correction * self.battery.capacity_design_c
            total_old = self.battery.y1 + self.battery.y2
            total_new = max(0, min(total_old + delta_q, self.battery.capacity_design_c))
            
            if total_old > 1e-9:
                ratio = total_new / total_old
                self.battery.y1 *= ratio
                self.battery.y2 *= ratio

        # --- B. R 内阻学习 ---
        if current > 0.5:
            r_update = - self.lr_r * voltage_error * abs(current)
            self.battery.R_base += r_update
            self.battery.R_base = max(0.01, min(self.battery.R_base, 2.0))

        # 记录
        self.logs['r_history'].append(self.battery.R_base)
        self.logs['soc_correction'].append(soc_adjust)

    def check_full_charge_calibration(self, v_current):
        """
        新增：充电末端校准逻辑
        (将原来写在循环里的 if > 4.15 逻辑封装进来)
        """
        if v_current > 4.15:
            # 软校准：让当前电量向满电逼近 50%
            current_q = self.battery.y1 + self.battery.y2
            target_q = self.battery.capacity_design_c
            
            # 融合公式：新电量 = 0.5 * 旧电量 + 0.5 * 满电
            new_q = current_q * 0.5 + target_q * 0.5
            
            if current_q > 0:
                ratio = new_q / current_q
                self.battery.y1 *= ratio
                self.battery.y2 *= ratio
            return True
        return False