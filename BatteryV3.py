import numpy as np
from scipy.optimize import brentq

# ==========================================
# 物理常数定义
# ==========================================
FARADAY_CONST = 96485.3329  # s A / mol
GAS_CONST = 8.31446         # J / (mol K)


class PhysicsBattery:
    """
    基于物理机理的增强型锂离子电池模型
    """

    
    def __init__(self, design_capacity_ah=2.02706564, initial_soc=0.94314107, initial_temp_c=25.0):
        # --- 1. 基础电气参数 ---
        self.E0 = 3.95445374
        self.R_base = 0.20222941
        self.K = 0.11498637
        self.A_vol = 0.60987048
        self.B_vol = 3.18761375
        
        # --- 2. KiBaM 动力学参数 ---
        self.c = 0.65
        self.k_diff_ref = 0.05 
        self.Ea_diff = 20000.0
        
        # --- 3. SEI 生长与老化参数 ---
        self.M_sei = 0.162
        self.rho_sei = 1690.0
        self.kappa_sei = 5e-6
        self.U_s_ref = 0.4
        self.D_sol_ref = 2.5e-22 
        self.Ea_sol = 50000.0
        self.C_sol_bulk = 4500.0
        self.A_surf = 0.15
        
        # --- 4. 热模型参数 ---
        self.mass = 0.045
        self.Cp = 1100.0
        self.h_conv = 5.0
        self.A_cool = 0.008
        self.T_ref = 298.15
        self.V_cutoff = 2.0  # 截止电压
        
        # --- 状态变量初始化 ---
        self.design_capacity_ah = design_capacity_ah
        self.capacity_design_c = design_capacity_ah * 3600.0
        self.q_total = self.capacity_design_c * initial_soc
        
        self.y1 = self.q_total * self.c
        self.y2 = self.q_total * (1 - self.c)
        self.temp_k = initial_temp_c + 273.15
        
        self.L_sei = 5e-9
        self.q_loss_acc = 0.0
        self.R_sei = 0.0

        # 记录历史数据
        self.history = {
            'time': [], 'voltage': [], 'current': [], 'temp': [], 
            'soc': [], 'soh': [], 'sei_thickness': [], 'heat_rev': [], 'heat_irr': []
        }

    # ------------------------------------------
    # 辅助函数
    # ------------------------------------------
    def _get_arrhenius_factor(self, Ea):
        return np.exp((Ea / GAS_CONST) * (1.0/self.T_ref - 1.0/self.temp_k))

    def _get_entropic_coefficient(self, soc):
        du_dt = (-0.0004 + 0.0040 * soc - 0.0150 * soc**2 + 
                 0.0250 * soc**3 - 0.0180 * soc**4 + 0.0044 * soc**5)
        return np.clip(du_dt, -0.05, 0.05)  # 限制范围，防止异常值


    def get_displayed_soc(self, current_v):
        # 模拟 BMS 的映射逻辑
        # 比如：3.4V 对应显示的 0%，4.2V 对应显示的 100%
        v_min, v_max = 3.4, 4.2
        displayed = (current_v - v_min) / (v_max - v_min)
        return np.clip(displayed, 0, 1.0)

    # ------------------------------------------
    # 校准接口
    # ------------------------------------------
    def calibrate_state(self, observed_val, mode='soc', current_a=0.0):
        if mode == 'soc':
            real_soc = np.clip(observed_val, 0.0, 1.0)
            capacity_actual_c = self.capacity_design_c - self.q_loss_acc
            new_q_total = capacity_actual_c * real_soc
            self.y1 = new_q_total * self.c
            self.y2 = new_q_total * (1 - self.c)
            print(f"系统校准: SOC 修正为 {real_soc*100:.2f}%")
            
        elif mode == 'voltage':
            target_voltage = observed_val
            
            def voltage_error(test_soc):
                test_soc = np.clip(test_soc, 0.01, 0.99)
                capacity_actual_c = self.capacity_design_c - self.q_loss_acc
                q_discharged_ah = capacity_actual_c * (1 - test_soc) / 3600.0
                term_pol = self.K * (self.design_capacity_ah / max(1e-3, self.design_capacity_ah - q_discharged_ah))
                term_exp = self.A_vol * np.exp(-self.B_vol * q_discharged_ah)
                ocv = self.E0 - term_pol + term_exp
                r_temp_factor = np.exp(1000 * (1.0/self.temp_k - 1.0/self.T_ref))
                r_total = self.R_base * r_temp_factor + self.R_sei
                est_voltage = ocv - current_a * r_total
                return est_voltage - target_voltage

            try:
                calibrated_soc = brentq(voltage_error, 0.01, 0.99)
                capacity_actual_c = self.capacity_design_c - self.q_loss_acc
                new_q_total = capacity_actual_c * calibrated_soc
                self.y1 = new_q_total * self.c
                self.y2 = new_q_total * (1 - self.c)
                print(f"系统校准: 电压 {target_voltage:.3f}V -> 反推 SOC {calibrated_soc*100:.2f}%")
            except ValueError:
                print("校准失败: 观测电压超出模型可达范围")

    # ------------------------------------------
    # 步进更新
    # ------------------------------------------
    def step(self, current_a, dt, temp_env_c=25.0):
        temp_env_k = temp_env_c + 273.15

        capacity_actual_c = max(self.capacity_design_c - self.q_loss_acc, 1.0) # 防止除零

        # --- 2. KiBaM 电荷更新：子步迭代防止数值爆炸 ---
        n_substeps = 10  
        dt_sub = dt / n_substeps
        
        for _ in range(n_substeps):
            k_diff_curr = self.k_diff_ref * self._get_arrhenius_factor(self.Ea_diff)
            
            # 计算当前水位高度 (h1, h2)
            h1 = self.y1 / (self.c * capacity_actual_c + 1e-9)
            h2 = self.y2 / ((1 - self.c) * capacity_actual_c + 1e-9)
            
            # 内部扩散电流 (从槽2流向槽1)
            i_diff = k_diff_curr * (h2 - h1) * capacity_actual_c
            
            # 状态演化
            self.y1 -= (current_a - i_diff) * dt_sub
            self.y2 -= i_diff * dt_sub
            
            # 修正后的剪切逻辑：只限制总电量，允许 y1/y2 比例动态偏离
            q_total_current = self.y1 + self.y2
            if q_total_current > capacity_actual_c:
                factor = capacity_actual_c / q_total_current
                self.y1 *= factor
                self.y2 *= factor
            elif q_total_current < 0:
                self.y1, self.y2 = 0.0, 0.0
        
        # 计算该步骤结束后的 SOC
        soc = (self.y1 + self.y2) / (capacity_actual_c + 1e-9)

        # --- 3. SEI 老化：随时间线性演化 ---
        D_sol_curr = self.D_sol_ref * self._get_arrhenius_factor(self.Ea_sol)
        j_side = (FARADAY_CONST * D_sol_curr * self.C_sol_bulk) / max(self.L_sei, 1e-12)
        dL_dt = (j_side * self.M_sei) / (self.rho_sei * FARADAY_CONST)
        self.L_sei += np.clip(dL_dt, 0, 1e-10) * dt
        
        i_side_total = j_side * self.A_surf
        self.q_loss_acc += i_side_total * dt
        self.R_sei = self.L_sei / (self.kappa_sei * self.A_surf)

        # --- 4. 热模型：考虑可逆与不可逆热 ---
        r_temp_factor = np.exp(1000 * (1.0/self.temp_k - 1.0/self.T_ref))
        r_total = self.R_base * r_temp_factor + self.R_sei
        
        q_irr = (current_a ** 2) * r_total
        du_dt = self._get_entropic_coefficient(soc)
        q_rev = current_a * self.temp_k * du_dt
        q_gen = q_irr + q_rev
        q_diss = self.h_conv * self.A_cool * (self.temp_k - temp_env_k)
        
        # 温度更新
        self.temp_k += (q_gen - q_diss) / (self.mass * self.Cp) * dt

        # --- 5. 电压计算 ---
        q_discharged_ah = self.design_capacity_ah - (self.y1 + self.y2) / 3600.0
        remaining_cap = max(1e-4, self.design_capacity_ah - q_discharged_ah) # 加上 1e-4 防止除零
    
        # Shepherd 方程计算 OCV (保持原公式)
        ocv = self.E0 - self.K * (self.design_capacity_ah / remaining_cap) \
            + self.A_vol * np.exp(-self.B_vol * q_discharged_ah)
        
        # 计算端电压
        volt_term = ocv - current_a * r_total
        
        # 可选：只在最后输出时做一个极端的物理限制（比如不能低于 0V），
        # 但在拟合过程中，让它掉下去更有利于梯度下降找到正确的容量。
        if volt_term < 0.1: 
            volt_term = 0.1

        # --- 6. 数据记录 ---
        # (确保 self.history 字典已在 __init__ 中初始化)
        current_time = self.history['time'][-1] + dt if self.history['time'] else 0
        self.history['time'].append(current_time)
        self.history['voltage'].append(volt_term)
        self.history['current'].append(current_a)
        self.history['temp'].append(self.temp_k - 273.15)
        self.history['soc'].append(soc)
        self.history['soh'].append(capacity_actual_c / self.capacity_design_c)
        self.history['sei_thickness'].append(self.L_sei)
        self.history['heat_rev'].append(q_rev)
        self.history['heat_irr'].append(q_irr)
        
        return volt_term, soc