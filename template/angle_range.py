import numpy as np

def compute_angle_range(theta_ab, theta_bc):
    # 将角度转换为弧度
    theta_ab_rad = np.radians(theta_ab)
    theta_bc_rad = np.radians(theta_bc)

    # 计算夹角的余弦值
    cos_theta_ab = np.cos(theta_ab_rad)
    cos_theta_bc = np.cos(theta_bc_rad)

    # 检查夹角是否合法
    if abs(cos_theta_ab) > 1 or abs(cos_theta_bc) > 1:
        return None  # 无法确定夹角范围

    # 计算第三个夹角的范围
    a_rad = np.arccos(cos_theta_ab * cos_theta_bc - np.sin(theta_ab_rad) * np.sin(theta_bc_rad))
    b_rad = np.arccos(cos_theta_ab * cos_theta_bc + np.sin(theta_ab_rad) * np.sin(theta_bc_rad))

    # 将弧度转换为角度
    a_deg = np.degrees(a_rad)
    b_deg = np.degrees(b_rad)

    return [a_deg, b_deg]

# 示例使用
theta_ab = 30  # 第一个夹角的度数
theta_bc = 45  # 第二个夹角的度数

angle_range = compute_angle_range(theta_ab, theta_bc)
if angle_range is None:
    print("无法确定第三个夹角的范围")
else:
    print("第三个夹角的大小范围：[{}°, {}°]".format(angle_range[0], angle_range[1]))
