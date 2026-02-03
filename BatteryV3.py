import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ==========================================
# 1. 物理电池模型 (PhysicsBattery) - 2026 MCM A 专用版
# ==========================================
FARADAY_CONST = 96485.3329
GAS_CONST = 8.31446




class PhysicsBattery:
    def __init__(self, design_capacity_ah=1.85027448, initial_soc=1.0, initial_temp_c=25.0):
        # --- 基础电气参数 ---
        self.E0 = 4.31451447
        self.R_base_new = 0.031566936  # 新电池的基础欧姆内阻
        self.R_base = self.R_base_new 
        self.K = 0.0865715
        self.A_vol = 0.48140649
        self.B_vol = 2.44023213
        
        # --- KiBaM 动力学参数 ---
        self.c = 0.65
        self.k_diff_ref = 0.05 
        self.Ea_diff = 20000.0
        
        # --- SEI 与老化物理参数 (Physics-based Aging) ---
        # 参考文献：单粒子模型(SPM)中的副反应参数
        self.M_sei = 0.162       # SEI 摩尔质量 (kg/mol)
        self.rho_sei = 1690.0    # SEI 密度 (kg/m^3)
        self.kappa_sei = 5e-6    # SEI 电导率 (S/m) - 影响阻抗增加
        
        # 溶剂扩散参数 - 控制日历老化速度
        self.D_sol_ref = 2.5e-22 # 参考扩散系数 (m^2/s)
        self.Ea_sol = 50000.0    # 扩散活化能 (J/mol)
        self.C_sol_bulk = 4500.0 # 电解液溶剂浓度 (mol/m^3)
        self.A_surf = 0.15       # 有效电化学活性面积 (m^2)
        
        # --- 热模型 ---
        self.mass = 0.065 
        self.Cp = 1100.0
        self.h_conv = 8.0 
        self.A_cool = 0.012 
        self.T_ref = 298.15
        
        # --- 状态初始化 ---
        self.design_capacity_ah = design_capacity_ah
        self.capacity_design_c = design_capacity_ah * 3600.0
        
        # 老化状态变量
        self.L_sei = 5e-9        # SEI 初始厚度 (5nm)
        self.q_loss_acc = 0.0    # 累积容量损失 (Coulombs)
        self.R_sei = 0.0         # SEI 引起的额外内阻 (Ohms)
        
        # 初始电荷分配
        capacity_actual_c = self.capacity_design_c - self.q_loss_acc
        self.q_total = capacity_actual_c * initial_soc
        self.y1 = self.q_total * self.c
        self.y2 = self.q_total * (1 - self.c)
        
        self.temp_k = initial_temp_c + 273.15
        
        # 记录
        self.history = {'time': [], 'voltage': [], 'current': [], 'temp': [], 'soc': [], 'R_internal': []}

    def _get_arrhenius_factor(self, Ea):
        """计算温度对反应速率的影响"""
        return np.exp((Ea / GAS_CONST) * (1.0/self.T_ref - 1.0/self.temp_k))

    def _get_entropic_coefficient(self, soc):
        """熵热系数 dU/dT (V/K)"""
        du_dt = (-0.0004 + 0.0040 * soc - 0.0150 * soc**2 + 
                 0.0250 * soc**3 - 0.0180 * soc**4 + 0.0044 * soc**5)
        return np.clip(du_dt, -0.05, 0.05)
    
    # ==========================================
    # 【新增】老化计算逻辑 (SEI Growth)
    # ==========================================
    def _update_aging(self, dt, soc):
        """
        根据物理模型计算 SEI 膜增厚和锂离子损耗
        """
        # 1. 溶剂扩散系数 (受温度严重影响)
        # 高温下老化显著加速 (Arrhenius)
        D_sol = self.D_sol_ref * np.exp((self.Ea_sol / GAS_CONST) * (1.0/self.T_ref - 1.0/self.temp_k))
        
        # 2. 副反应电流 (Diffusion Limited)
        # 假设膜越厚，溶剂越难渗透，生长越慢 (自限性)
        # I_side 是消耗锂的电流 (A)，通常极小 (nA~uA 级别)
        # 增加 SOC 因子：高电量(SOC>80%)时阳极电位更低，副反应更剧烈 (经验修正)
        soc_stress_factor = 1.0 + 2.0 * (soc ** 2) 
        i_side_amp = (FARADAY_CONST * self.A_surf * D_sol * self.C_sol_bulk / self.L_sei) * soc_stress_factor
        
        # 3. 更新 SEI 厚度 (L_sei)
        # dL/dt = (i_side * M) / (n * F * rho * A)
        # n=1 (Li+ + e- -> Li)
        dL = (i_side_amp * self.M_sei * dt) / (1 * FARADAY_CONST * self.rho_sei * self.A_surf)
        self.L_sei += dL
        
        # 4. 更新容量损失 (Capacity Fade)
        # 消耗的锂离子电量
        dq_loss = i_side_amp * dt
        self.q_loss_acc += dq_loss
        
        # 5. 更新 SEI 内阻 (Power Fade)
        # R = L / (sigma * A)
        self.R_sei = self.L_sei / (self.kappa_sei * self.A_surf)

    def fast_forward_aging(self, years=2.0, avg_temp_c=25.0, avg_soc=0.8):
        """
        【赛题神器】一键将电池老化到 N 年后的状态
        用于对比 "New Phone" vs "2-Year Old Phone"
        """
        seconds = years * 365 * 24 * 3600
        print(f"--- Fast Forwarding Aging: {years} Years ---")
        
        # 临时覆盖温度和SOC进行估算
        original_temp = self.temp_k
        self.temp_k = avg_temp_c + 273.15
        
        # 为了加速，我们不跑 step 循环，而是直接积分解析解
        # 简化版抛物线生长定律: L(t) = sqrt(L0^2 + 2*k*t)
        # 这里为了保持逻辑一致，我们用大步长模拟
        
        sim_dt = 3600 * 24 # 每次跳一天
        steps = int(seconds / sim_dt)
        
        for _ in range(steps):
            self._update_aging(sim_dt, avg_soc)
            
        self.temp_k = original_temp # 恢复温度
        
        # 重新分配当前容量
        capacity_actual = self.capacity_design_c - self.q_loss_acc
        print(f"  > SEI Thickness: {self.L_sei*1e9:.2f} nm (Initial: 5.00 nm)")
        print(f"  > Capacity Loss: {self.q_loss_acc/3600:.3f} Ah ({(self.q_loss_acc/self.capacity_design_c)*100:.1f}%)")
        print(f"  > Resistance Increase: +{self.R_sei*1000:.2f} mOhm")
        print(f"  > Current Health (SOH): {(capacity_actual/self.capacity_design_c)*100:.1f}%")
        print("------------------------------------------")

    def calibrate_state(self, observed_val, mode='soc', current_a=0.0):
        # ... (保持原有的校准逻辑不变) ...
        # (为了节省篇幅，此处省略，请保留原来代码中的 calibrate_state 内容)
        pass 

    def step(self, current_a, dt, temp_env_c=25.0):
        temp_env_k = temp_env_c + 273.15
        
        # 1. 计算当前真实容量 (考虑老化)
        capacity_actual_c = max(self.capacity_design_c - self.q_loss_acc, 1.0)

        # 2. KiBaM 动力学模型 (电荷传输)
        n_substeps = 10  
        dt_sub = dt / n_substeps
        for _ in range(n_substeps):
            k_diff_curr = self.k_diff_ref * self._get_arrhenius_factor(self.Ea_diff)
            h1 = self.y1 / (self.c * capacity_actual_c + 1e-9)
            h2 = self.y2 / ((1 - self.c) * capacity_actual_c + 1e-9)
            i_diff = k_diff_curr * (h2 - h1) * capacity_actual_c
            self.y1 -= (current_a - i_diff) * dt_sub
            self.y2 -= i_diff * dt_sub
            
            # 边界保护
            q_now = self.y1 + self.y2
            if q_now > capacity_actual_c: 
                factor = capacity_actual_c/q_now
                self.y1 *= factor; self.y2 *= factor
            elif q_now < 0: self.y1 = 0; self.y2 = 0
        
        # 计算 SOC
        soc = (self.y1 + self.y2) / (capacity_actual_c + 1e-9)

        # 3. 【新增】调用老化模型
        # 在充放电过程中实时计算 SEI 增长
        self._update_aging(dt, soc)

        # 4. 热模型与内阻
        r_temp_factor = np.exp(1000 * (1.0/self.temp_k - 1.0/self.T_ref))
        
        # 总内阻 = 基础内阻(受温度影响) + SEI老化内阻
        r_total = self.R_base * r_temp_factor + self.R_sei
        
        q_gen = (current_a**2)*r_total + current_a*self.temp_k*self._get_entropic_coefficient(soc)
        q_diss = self.h_conv * self.A_cool * (self.temp_k - temp_env_k)
        self.temp_k += (q_gen - q_diss) / (self.mass * self.Cp) * dt

        # 5. 电压计算
        q_discharged_ah = (self.design_capacity_ah * 3600 - (self.y1 + self.y2)) / 3600.0
        # 注意：Shepherd 模型通常基于设计容量计算 OCV 曲线形状，但内阻压降会变大
        remaining_cap = max(1e-4, self.design_capacity_ah - q_discharged_ah)
        
        ocv = self.E0 - self.K * (self.design_capacity_ah / remaining_cap) + \
              self.A_vol * np.exp(-self.B_vol * q_discharged_ah)
        
        volt_term = ocv - current_a * r_total
        
        # 记录数据
        curr_t = self.history['time'][-1] + dt if self.history['time'] else 0
        self.history['time'].append(curr_t)
        self.history['voltage'].append(volt_term)
        self.history['current'].append(current_a)
        self.history['temp'].append(self.temp_k - 273.15)
        self.history['soc'].append(soc)
        self.history['R_internal'].append(r_total) # 记录内阻变化
        
        return volt_term, soc