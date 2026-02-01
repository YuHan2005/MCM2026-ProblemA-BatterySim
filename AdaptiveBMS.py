import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from BatteryV3 import PhysicsBattery

# ==========================================
# 1. 修正后的 BMS 类 (参数与逻辑完全对齐)
# ==========================================
class AdaptiveBMS:
    def __init__(self, battery_model, learning_rate_R=5e-5, feedback_gain_soc=0.005):
        """
        修正点：默认 learning_rate_R 提高到 5e-5，与你手动运行的参数一致
        """
        self.battery = battery_model
        self.lr_r = learning_rate_R
        self.gain_soc = feedback_gain_soc   
        self.logs = {'r_history': [], 'soc_correction': []}

    def update(self, current, dt, temp_env_c,v_meas):
         
        v_est, soc_est = self.battery.step(current, dt, temp_env_c=temp_env_c)
        
        voltage_error = v_meas - v_est

        # ---------------------------------------------------------
        # 闭环策略 A: 软性 SOC 修正 (Soft SOC Nudging)
        # ---------------------------------------------------------
        # 如果模型电压 < 真实电压，说明模型 SOC 估低了，需要加一点
        # 如果模型电压 > 真实电压，说明模型 SOC 估高了，需要减一点
        # 只有在电流较大时(电压包含SOC信息)且非剧烈变化时才修正
        if abs(current) > 0.1: 
            # 增益 K 需要非常小，否则会震荡
            # 反馈量 = 增益 * 电压误差 * dt
            # 注意符号：SOC高 -> OCV高 -> V_est高 -> Error负 -> SOC应减小。方向正确。
            
            soc_correction = self.gain_soc * voltage_error * dt
            
            # 将修正量应用回模型的内部状态 (y1, y2)
            # 按比例分配给 bound/available charge
            total_charge_old = self.battery.y1 + self.battery.y2
            total_charge_new = total_charge_old + (soc_correction * self.battery.capacity_design_c)
            
            # 防止修正出界
            total_charge_new = max(0, min(total_charge_new, self.battery.capacity_design_c))
            
            if total_charge_old > 0:
                ratio = total_charge_new / total_charge_old
                self.battery.y1 *= ratio
                self.battery.y2 *= ratio
        
        # ---------------------------------------------------------
        # 闭环策略 B: 在线内阻学习 (Online R Learning)
        # ---------------------------------------------------------
        # 简单的 LMS (Least Mean Squares) 梯度下降
        # 如果 V_est 总是比 V_meas 大 (预测偏高)，说明还需要更大的压降 -> R 需要变大
        # 只有在放电时更新 R
        if current > 0.5: 
            # 梯度方向：Error = V_meas - (OCV - I*R)
            # d(Error)/dR = I
            # R_new = R_old + alpha * Error * I
            # 这里的符号：如果 V_meas < V_est (Error < 0)，说明真实掉电快，内阻其实更大
            # 所以我们要 增加 R。
            # update = - learning_rate * error * current
            
            r_update = - self.lr_r * voltage_error * abs(current)
            self.battery.R_base += r_update
            
            # 物理约束：内阻不能是负的，也不能无限大
            self.battery.R_base = max(0.01, min(self.battery.R_base, 2.0))

        return v_est, soc_est
    

    def set_lr(self,lr):
        self.lr_r = lr

