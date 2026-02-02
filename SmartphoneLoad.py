# ==========================================
# 3. 智能手机负载模型 (SmartphoneLoad)
# ==========================================
class SmartphoneLoad:
    def __init__(self, device_type="flagship_2025"):
        # 功耗参数 (W) - 已针对 2026 旗舰机微调
        self.P_IDLE_BASE = 0.05      # 亮屏前的基础活跃功耗
        self.P_DEEP_SLEEP = 0.008    # 深度睡眠 (8mW)
        
        self.P_SCREEN_MAX = 2.0      # 屏幕满亮度 (平均值，考虑OLED非全白)
        self.P_SCREEN_BASE = 0.0     
        self.P_CPU_IDLE = 0.2
        self.P_CPU_MAX = 5.0         # 满载功耗
        
        self.P_WIFI_ACTIVE = 0.3
        self.P_5G_ACTIVE = 1.5       # 5G 依然耗电
        self.P_4G_ACTIVE = 1.0
        self.P_NET_STANDBY = 0.02
        self.P_GPS_ACTIVE = 0.3
        self.P_AUDIO = 0.3           # DSP 音频解码，功耗很低

    def get_current_demand(self, voltage_v, state):
        voltage_v = max(voltage_v, 2.5) # 防止除零
        
        # --- 1. 深度休眠逻辑 ---
        # 条件：屏幕关 + CPU低 + 网络无吞吐
        if not state.get('screen_on', False) and state.get('cpu_load', 0) < 0.02 and state.get('network_throughput', 0) < 0.01:
            power = self.P_DEEP_SLEEP
        else:
            # --- 2. 正常活跃模式 ---
            power = self.P_IDLE_BASE
            
            # 屏幕
            if state.get('screen_on', False):
                power += self.P_SCREEN_BASE + self.P_SCREEN_MAX * state.get('screen_brightness', 0.5)
            
            # CPU
            cpu = state.get('cpu_load', 0.0)
            power += self.P_CPU_IDLE + (self.P_CPU_MAX - self.P_CPU_IDLE) * (cpu ** 1.5)
            
            # 网络
            net_type = state.get('network_type', 'none')
            net_load = state.get('network_throughput', 0.0)
            if net_type == 'wifi': power += self.P_NET_STANDBY + self.P_WIFI_ACTIVE * net_load
            elif net_type == '5g': power += self.P_NET_STANDBY + self.P_5G_ACTIVE * net_load
            elif net_type == '4g': power += self.P_NET_STANDBY + self.P_4G_ACTIVE * net_load
            
            # 其他
            if state.get('gps_on', False): power += self.P_GPS_ACTIVE
            if state.get('audio_on', False): power += self.P_AUDIO

        # 计算电流 (I = P/V)，假设 DC-DC 效率 95%
        return power / (voltage_v * 0.95)