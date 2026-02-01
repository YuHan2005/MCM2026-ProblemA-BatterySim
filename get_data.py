import pandas as pd
import os
import shutil

# ==========================================
# 1. 路径与参数定义 (使用 raw string r'' 防止转义问题)
# ==========================================
# 注意：Windows 路径建议在字符串前加 r，或者使用双反斜杠 \\
data_path = r"C:\Users\lenovo\Desktop\archive\cleaned_dataset\data"
save_path = r"C:\Users\lenovo\Desktop\archive\cleaned_dataset\charge"
battery_id = "B0047"
a_type = "charge"

# ==========================================
# 2. 读取元数据并筛选
# ==========================================
# 读取 metadata.csv
df_meta = pd.read_csv(r'C:\Users\lenovo\Desktop\archive\cleaned_dataset\metadata.csv')

# 筛选符合条件的文件名
# 假设 metadata.csv 中的顺序就是时间顺序，如果不确定，建议按 'start_time' 排序
# df_filtered = df_meta.loc[(df_meta['battery_id'] == battery_id) & (df_meta['type'] == a_type)].sort_values('start_time')
data_fils = df_meta.loc[(df_meta['battery_id'] == battery_id) & (df_meta['type'] == a_type), 'filename']

print(f"找到 {len(data_fils)} 个符合条件的文件。")

# ==========================================
# 3. 创建目标文件夹 (如果不存在)
# ==========================================
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"已创建目录: {save_path}")
else:
    print(f"目录已存在: {save_path}")

# ==========================================
# 4. 按顺序复制并重命名
# ==========================================
count = 0
# enumerate 用于同时获取索引(从0开始)和文件名
for i, filename in enumerate(data_fils):
    # 构造源文件完整路径
    src_file = os.path.join(data_path, filename)
    
    # 构造目标文件名 (按顺序命名为 1.csv, 2.csv ...)
    new_filename = f"{i + 1}.csv"
    dst_file = os.path.join(save_path, new_filename)
    
    try:
        # 检查源文件是否存在
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            count += 1
            if count % 10 == 0: # 每复制10个打印一次，避免刷屏
                print(f"进度: 已复制 {count} 个文件 -> {new_filename}")
        else:
            print(f"[警告] 源文件缺失: {filename}")
            
    except Exception as e:
        print(f"[错误] 复制 {filename} 时出错: {e}")

print(f"\n处理完成！共成功复制 {count} 个文件到:\n{save_path}")