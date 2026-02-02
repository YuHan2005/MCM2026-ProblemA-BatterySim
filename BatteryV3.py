import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ==========================================
# 1. 物理电池模型 (PhysicsBattery)
# ==========================================
FARADAY_CONST = 96485.3329
GAS_CONST = 8.31446

class PhysicsBattery:
    def __init__(self, design_capacity_ah=4.8, initial_soc=1.0, initial_temp_c=25.0):
        # 修正：默认容量设为 4.8Ah (2026年主流旗舰标准)
        
        # --- 基础电气参数 ---
        self.E0 = 3.95445374
        # 修正：大容量电池内阻通常更低 (模拟双电芯并联，R从0.2降为0.08)
        self.R_base = 0.08  
        self.K = 0.11498637
        self.A_vol = 0.60987048
        self.B_vol = 3.18761375
        
        # --- KiBaM 动力学参数 ---
        self.c = 0.65
        self.k_diff_ref = 0.05 
        self.Ea_diff = 20000.0
        
        # --- SEI 与老化 ---
        self.M_sei = 0.162
        self.rho_sei = 1690.0
        self.kappa_sei = 5e-6
        self.U_s_ref = 0.4
        self.D_sol_ref = 2.5e-22 
        self.Ea_sol = 50000.0
        self.C_sol_bulk = 4500.0
        self.A_surf = 0.15
        
        # --- 热模型 ---
        self.mass = 0.065 # 电池变大，质量增加
        self.Cp = 1100.0
        self.h_conv = 8.0 # 散热系数略微调整
        self.A_cool = 0.012 # 表面积增加
        self.T_ref = 298.15
        self.V_cutoff = 3.0 # 提高保护电压至3.0V (保护电池)
        
        # --- 状态初始化 ---
        self.design_capacity_ah = design_capacity_ah
        self.capacity_design_c = design_capacity_ah * 3600.0
        self.q_total = self.capacity_design_c * initial_soc
        
        self.y1 = self.q_total * self.c
        self.y2 = self.q_total * (1 - self.c)
        self.temp_k = initial_temp_c + 273.15
        
        self.L_sei = 5e-9
        self.q_loss_acc = 0.0
        self.R_sei = 0.0

        self.history = {'time': [], 'voltage': [], 'current': [], 'temp': [], 'soc': []}

    def _get_arrhenius_factor(self, Ea):
        return np.exp((Ea / GAS_CONST) * (1.0/self.T_ref - 1.0/self.temp_k))

    def _get_entropic_coefficient(self, soc):
        du_dt = (-0.0004 + 0.0040 * soc - 0.0150 * soc**2 + 
                 0.0250 * soc**3 - 0.0180 * soc**4 + 0.0044 * soc**5)
        return np.clip(du_dt, -0.05, 0.05)

    # ==========================================
    # 【重点】这就是你缺失的校准函数
    # ==========================================
    def calibrate_state(self, observed_val, mode='soc', current_a=0.0):
        """
        强制校准电池内部状态，使其符合观测到的电压或SOC
        """
        if mode == 'soc':
            real_soc = np.clip(observed_val, 0.0, 1.0)
            capacity_actual_c = self.capacity_design_c - self.q_loss_acc
            new_q_total = capacity_actual_c * real_soc
            self.y1 = new_q_total * self.c
            self.y2 = new_q_total * (1 - self.c)
            # print(f"系统校准: SOC 修正为 {real_soc*100:.2f}%")
            
        elif mode == 'voltage':
            target_voltage = observed_val
            
            # 定义误差函数：给定 SOC，计算出来的电压与目标电压的差
            def voltage_error(test_soc):
                test_soc = np.clip(test_soc, 0.01, 0.99)
                capacity_actual_c = self.capacity_design_c - self.q_loss_acc
                q_discharged_ah = capacity_actual_c * (1 - test_soc) / 3600.0
                
                # Shepherd 模型计算 OCV
                remaining_cap = max(1e-3, self.design_capacity_ah - q_discharged_ah)
                term_pol = self.K * (self.design_capacity_ah / remaining_cap)
                term_exp = self.A_vol * np.exp(-self.B_vol * q_discharged_ah)
                ocv = self.E0 - term_pol + term_exp
                
                # 考虑热和内阻
                r_temp_factor = np.exp(1000 * (1.0/self.temp_k - 1.0/self.T_ref))
                r_total = self.R_base * r_temp_factor + self.R_sei
                est_voltage = ocv - current_a * r_total
                
                return est_voltage - target_voltage

            try:
                # 使用求根公式反推 SOC
                calibrated_soc = brentq(voltage_error, 0.01, 0.999)
                
                # 更新内部电荷状态
                capacity_actual_c = self.capacity_design_c - self.q_loss_acc
                new_q_total = capacity_actual_c * calibrated_soc
                self.y1 = new_q_total * self.c
                self.y2 = new_q_total * (1 - self.c)
                print(f"  [Calibration Success] V_target={target_voltage:.3f}V -> SOC={calibrated_soc*100:.2f}%")
            except ValueError:
                print("  [Calibration Warning] 观测电压超出模型范围，跳过校准。")

    def step(self, current_a, dt, temp_env_c=25.0):
        temp_env_k = temp_env_c + 273.15
        capacity_actual_c = max(self.capacity_design_c - self.q_loss_acc, 1.0)

        # KiBaM (子步计算)
        n_substeps = 10  
        dt_sub = dt / n_substeps
        for _ in range(n_substeps):
            k_diff_curr = self.k_diff_ref * self._get_arrhenius_factor(self.Ea_diff)
            h1 = self.y1 / (self.c * capacity_actual_c + 1e-9)
            h2 = self.y2 / ((1 - self.c) * capacity_actual_c + 1e-9)
            i_diff = k_diff_curr * (h2 - h1) * capacity_actual_c
            self.y1 -= (current_a - i_diff) * dt_sub
            self.y2 -= i_diff * dt_sub
            
            # 边界修正
            q_now = self.y1 + self.y2
            if q_now > capacity_actual_c: 
                factor = capacity_actual_c/q_now
                self.y1 *= factor; self.y2 *= factor
            elif q_now < 0: self.y1 = 0; self.y2 = 0
        
        soc = (self.y1 + self.y2) / (capacity_actual_c + 1e-9)

        # SEI & 热
        r_temp_factor = np.exp(1000 * (1.0/self.temp_k - 1.0/self.T_ref))
        r_total = self.R_base * r_temp_factor + self.R_sei
        
        q_gen = (current_a**2)*r_total + current_a*self.temp_k*self._get_entropic_coefficient(soc)
        q_diss = self.h_conv * self.A_cool * (self.temp_k - temp_env_k)
        self.temp_k += (q_gen - q_diss) / (self.mass * self.Cp) * dt

        # 电压
        q_discharged_ah = self.design_capacity_ah - (self.y1 + self.y2) / 3600.0
        remaining_cap = max(1e-4, self.design_capacity_ah - q_discharged_ah)
        ocv = self.E0 - self.K * (self.design_capacity_ah / remaining_cap) + \
              self.A_vol * np.exp(-self.B_vol * q_discharged_ah)
        volt_term = max(0.1, ocv - current_a * r_total)

        # 记录
        curr_t = self.history['time'][-1] + dt if self.history['time'] else 0
        self.history['time'].append(curr_t)
        self.history['voltage'].append(volt_term)
        self.history['current'].append(current_a)
        self.history['temp'].append(self.temp_k - 273.15)
        self.history['soc'].append(soc)
        
        return volt_term, soc