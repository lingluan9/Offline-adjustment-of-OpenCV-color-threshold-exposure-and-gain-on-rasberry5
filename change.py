import cv2
import numpy as np
import json
import os
from gpiozero import Button
import time
import sys

# 定义按键对应的GPIO引脚（BCM编号）
BUTTON_PINS = {
    2: "Button1",
    3: "Button2",
    4: "Button3",
    17: "Button4"
}
# 创建按键对象（启用内部上拉）
buttons = {pin: Button(pin, pull_up=True) for pin in BUTTON_PINS.keys()}

# 按键状态管理
class ButtonState:
    def __init__(self):
        self.current_state = None  # 当前物理状态
        self.last_state = None     # 上一次物理状态
        self.last_debounce_time = 0  # 上次状态变化时间
        self.event_triggered = False  # 事件是否已触发
        self.press_detected = False   # 按下是否已检测
    
    def update(self, current_value, current_time):
        # 状态发生变化
        if current_value != self.last_state:
            self.last_debounce_time = current_time
        
        # 检查是否超过消抖时间（20毫秒）
        if (current_time - self.last_debounce_time) > 0.02:
            # 状态稳定
            if current_value != self.current_state:
                self.current_state = current_value
                
                # 检测到按下（低电平）
                if self.current_state == 0 and not self.press_detected:
                    self.press_detected = True
                    self.event_triggered = False
                
                # 检测到释放（高电平）且按下已检测
                elif self.current_state == 1 and self.press_detected:
                    self.press_detected = False
                    self.event_triggered = True
                    return True  # 返回完整事件信号
        
        self.last_state = current_value
        return False

# 初始化按键状态跟踪
button_states = {pin: ButtonState() for pin in BUTTON_PINS.keys()}

# 按键变量
pagex=1
page2=0
save_nowHSV=0
save_currentHSV=0
load_eg=0
H_key=0
S_key=0
V_key=0
H_up_key=0
S_up_key=0
V_up_key=0
H_low_key=0
S_low_key=0
V_low_key=0
exp_key=0
gain_key=0
count_down=0
count_up=1
last_page=0

# 全局变量
current_exp=350
current_gain=7
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, current_exp)      # 曝光
cap.set(cv2.CAP_PROP_GAIN, current_gain)            # 增益
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# 方框参数
BOX_SIZE = 100  # 方框边长（像素）
box_color = (0, 255, 0)  # 方框颜色（绿色）
thickness = 2           # 方框线宽

# HSV范围存储 - 现在使用列表代替字典
current_min = [0, 0, 0]  # 当前使用的最小HSV值
current_max = [0, 0, 0]  # 当前使用的最大HSV值
file_path = "/home/wheeltec/opencv/hsv_range.json"

# 调整模式标志
h_low_fix = 0
h_up_fix = 0
s_low_fix = 0
s_up_fix = 0
v_low_fix = 0
v_up_fix = 0
exp_fix = 0
gain_fix = 0

def get_hsv_range(hsv_img, center_x, center_y, box_size):
    """获取中心方框区域的HSV范围"""
    half_size = box_size // 2
    x1, y1 = center_x - half_size, center_y - half_size
    x2, y2 = center_x + half_size, center_y + half_size
    
    # 确保不超出图像边界
    h, w = hsv_img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # 提取方框区域
    roi = hsv_img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None, None
    
    # 计算HSV通道的最小、最大和平均值
    h_min, h_max = np.min(roi[:,:,0]), np.max(roi[:,:,0])
    s_min, s_max = np.min(roi[:,:,1]), np.max(roi[:,:,1])
    v_min, v_max = np.min(roi[:,:,2]), np.max(roi[:,:,2])
    
    # 转换为Python原生类型
    min_val = [int(h_min), int(s_min), int(v_min)]
    max_val = [int(h_max), int(s_max), int(v_max)]
    
    return min_val, max_val

def save_hsv_range_to_file(min_val, max_val,exp , gain, filename):
    """保存HSV范围到JSON文件"""
    try:
        # 创建可序列化的字典
        save_data = {
            'min': min_val,
            'max': max_val,
            'exp':exp,
            'gain':gain
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"HSV范围已保存到 {filename}")
        print(f"Min: {min_val}, Max: {max_val}")
        print(f"exp: {exp}, gain: {gain}")
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def load_hsv_range_from_file(filename):
    """从JSON文件加载HSV范围"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # 确保数据格式正确
            if 'min' in data and 'max' in data:
                min_val = data['min']
                max_val = data['max']
                exp_val = data['exp']
                gain_val = data['gain']
                
                # 确保是整数列表
                min_val = [int(x) for x in min_val]
                max_val = [int(x) for x in max_val]
                exp_val = int(exp_val)
                gain_val = int(gain_val)
                
                print(f"从 {filename} 加载HSV范围成功")
                print(f"Min: {min_val}, Max: {max_val}")
                print(f"exp: {exp_val}, gain: {gain_val}")
                return min_val, max_val, exp_val, gain_val
            else:
                print("文件格式不正确")
                return None, None
        else:
            print(f"文件 {filename} 不存在")
            return None, None
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None

def merge_contours(contours, max_distance=100):
    # 计算所有边界矩形
    rects = [cv2.boundingRect(c) for c in contours]
    
    # 合并相近的矩形
    merged = []
    for rect in rects:
        x, y, w, h = rect
        found = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            # 计算两个矩形的中心距离
            center1 = (x + w/2, y + h/2)
            center2 = (mx + mw/2, my + mh/2)
            distance = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
            
            if distance < max_distance:
                # 合并两个矩形
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x+w, mx+mw) - new_x
                new_h = max(y+h, my+mh) - new_y
                merged[i] = (new_x, new_y, new_w, new_h)
                found = True
                break
        
        if not found:
            merged.append(rect)
    
    return merged

def color_detection(frame, min_val, max_val):
    if min_val is None or max_val is None:
        return frame
    
    lower1 = np.array(min_val)
    upper1 = np.array(max_val)
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_frame, lower1, upper1)
    
    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小面积轮廓
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    # 合并相近轮廓
    merged_rects = merge_contours(filtered_contours)
    
    # 绘制合并后的矩形
    for rect in merged_rects:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "saved_color", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    global current_min, current_max,current_exp, current_gain, h_low_fix, h_up_fix, s_low_fix, s_up_fix, v_low_fix, v_up_fix,exp_fix,gain_fix
    global pagex,page2,save_nowHSV,save_currentHSV,load_eg,H_up_key,S_up_key,V_up_key,H_low_key,S_low_key,V_low_key,exp_key,gain_key,H_key,S_key,V_key,count_down,count_up,last_page
    
    # 尝试加载保存的HSV范围
    loaded_min, loaded_max, loaded_exp, loaded_gain= load_hsv_range_from_file(file_path)
    if loaded_min and loaded_max:
        current_min = loaded_min
        current_max = loaded_max
        current_exp = loaded_exp
        current_gain = loaded_gain
    
    print("HSV范围检测与调整工具")
    print("=" * 50)
    print("按键说明:")
    print("  ESC    - 退出程序")
    print("  s      - 保存当前帧为图片")
    print("  w      - 保存当前HSV范围、曝光、增益到文件")
    print("  t      - 保存原始HSV范围、曝光、增益到文件")
    print("  r      - 从文件加载HSV范围、曝光、增益")
    print("  c      - 清除当前使用的HSV范围")
    print("  u      - 切换曝光调整模式")
    print("  h      - 切换增益调整模式")
    print("  q      - 加载曝光、增益")
    print("  j      - 切换H最小值调整模式")
    print("  i      - 切换H最大值调整模式")
    print("  k      - 切换S最小值调整模式")
    print("  o      - 切换S最大值调整模式")
    print("  l      - 切换V最小值调整模式")
    print("  p      - 切换V最大值调整模式")
    print("  a      - 减少当前选中的HSV值")
    print("  d      - 增加当前选中的HSV值")
    print("=" * 50)
    
    # 初始显示调整模式
    print("当前调整模式: 无")
    
    while True:
        ret, frame = cap.read()
        current_time = time.time()
        if not ret:
            print("错误: 无法从摄像头获取帧")
            break
            
        calibration = np.load("/home/wheeltec/opencv/camera_calibration_live.npz")
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(calibration['mtx'], calibration['dist'], (w, h), 1, (w, h))
        frame = cv2.undistort(frame, calibration['mtx'], calibration['dist'], None, newcameramtx)
        
        last_page=0
        for pin, name in BUTTON_PINS.items():
            # 获取当前物理值（0=按下，1=释放）
            current_value = 0 if buttons[pin].is_pressed else 1
            
            # 更新状态并检查是否触发事件
            if button_states[pin].update(current_value, current_time):
                print(f"{name} (GPIO{pin}): 完成一次按键动作")  #pin是引脚数字，2 3 4 17
                print(pagex)
                if(pagex==1 and last_page==0):
                    if(pin==2):
                        save_nowHSV=1
                    if(pin==3):
                        save_currentHSV=1
                    if(pin==4):
                        load_eg=1
                    if(pin==17):
                        pagex=2
                    last_page=1
                if(pagex==2 and page2==0 and last_page==0):
                    H_key=0
                    S_key=0
                    V_key=0
                    if(pin==2):
                        H_key=1
                        pagex=3
                    if(pin==3):
                        S_key=1
                        pagex=3
                    if(pin==4):
                        page2=1
                    if(pin==17):
                        pagex=1
                    last_page=1
                if(pagex==2 and page2==1 and last_page==0):
                    H_key=0
                    S_key=0
                    V_key=0
                    if(pin==2):
                        V_key=1
                        pagex=3
                    if(pin==3):
                        exp_key=1
                        pagex=4
                    if(pin==4):
                        gain_key=1
                        pagex=4
                    if(pin==17):
                        page2=0
                    last_page=1
                if(pagex==3 and last_page==0):
                    if(pin==2):
                        if(H_key==1):
                            H_low_key=1
                            pagex=4
                        if(S_key==1):
                            S_low_key=1
                            pagex=4
                        if(V_key==1):
                            V_low_key=1
                            pagex=4
                    if(pin==3):
                        if(H_key==1):
                            H_up_key=1
                            pagex=4
                        if(S_key==1):
                            S_up_key=1
                            pagex=4
                        if(V_key==1):
                            V_up_key=1
                            pagex=4
                    if(pin==4):
                        pagex=2
                    last_page=1
                if(pagex==4 and last_page==0):
                    if(pin==2):
                        count_down=1
                    if(pin==3):
                        count_up=1
                    if(pin==4):
                        pagex=2
                        if(h_low_fix==1):
                            H_low_key=1
                        if(h_up_fix==1):
                            H_up_key=1
                        if(s_low_fix==1):
                            S_low_key=1
                        if(s_up_fix==1):
                            S_up_key=1
                        if(v_low_fix==1):
                            V_low_key=1
                        if(v_up_fix==1):
                            V_up_key=1
                        if(exp_fix==1):
                            exp_key=1
                        if(gain_fix==1):
                            gain_key=1
                    last_page=1
        
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 获取图像中心坐标
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # 绘制中心方框
        half_size = BOX_SIZE // 2
        cv2.rectangle(frame, 
                     (center_x - half_size, center_y - half_size),
                     (center_x + half_size, center_y + half_size),
                     box_color, thickness)
        
        # 获取方框区域的HSV范围
        min_val, max_val = get_hsv_range(hsv_img, center_x, center_y, BOX_SIZE)
        
        # 显示当前HSV范围信息
        if min_val is not None and max_val is not None:
            # 显示实时HSV范围
            cv2.putText(frame, f"Real-time H: {min_val[0]}-{max_val[0]}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Real-time S: {min_val[1]}-{max_val[1]}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Real-time V: {min_val[2]}-{max_val[2]}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示标准HSV范围
            cv2.putText(frame, f"Real-time H(0-360): {min_val[0]*2}-{max_val[0]*2}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # 显示当前使用的HSV范围
        y_offset = 170
        cv2.putText(frame, f"Current H: {current_min[0]}-{current_max[0]}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Current S: {current_min[1]}-{current_max[1]}", 
                    (10, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Current V: {current_min[2]}-{current_max[2]}", 
                    (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Current H(0-360): {current_min[0]*2}-{current_max[0]*2}", 
                    (10, y_offset+90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Current exp: {current_exp} Current gain: {current_gain}", 
                    (10, y_offset+120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"pagex: {pagex}  page2: {page2}", 
                    (10, y_offset+150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 0), 2)
        
        # 显示调整模式
        mode_text = ""
        if h_low_fix: mode_text += "H_min "
        if h_up_fix: mode_text += "H_max "
        if s_low_fix: mode_text += "S_min "
        if s_up_fix: mode_text += "S_max "
        if v_low_fix: mode_text += "V_min "
        if v_up_fix: mode_text += "V_max "
        if exp_fix: mode_text += "exp "
        if gain_fix: mode_text += "gain "
        if not mode_text: mode_text = "none"
        
        cv2.putText(frame, f"调整模式: {mode_text}", (10, y_offset+180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 使用当前HSV范围进行颜色检测
        frame = color_detection(frame, current_min, current_max)
        
        # 显示操作提示
        cv2.putText(frame, "Press 'w': Save HSV Range, 'r': Load HSV Range", 
                    (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        cv2.putText(frame, "Press 'c': Clear Current Range, 's': Save Image", 
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        
        # 显示图像
        cv2.imshow("HSV Range Detection & Adjustment", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break
        elif key == ord('s'):  # 保存当前帧
            cv2.imwrite("hsv_detection_snapshot.jpg", frame)
            print("截图已保存为 'hsv_detection_snapshot.jpg'")
        elif save_nowHSV==1:  # 保存原始HSV范围到文件
            save_hsv_range_to_file(min_val, max_val,current_exp,current_gain, file_path)
            save_nowHSV=0
        elif save_currentHSV==1:  # 保存当前HSV范围到文件
            save_hsv_range_to_file(current_min, current_max,current_exp,current_gain,  file_path)
            save_currentHSV=0
        elif load_eg==1:  # 从文件加载HSV范围
            loaded_min, loaded_max, loaded_exp, loaded_gain = load_hsv_range_from_file(file_path)
            if loaded_min and loaded_max:
                current_min = loaded_min
                current_max = loaded_max
                current_exp = loaded_exp
                current_gain = loaded_gain
                cap.set(cv2.CAP_PROP_EXPOSURE, current_exp)      # 曝光
                cap.set(cv2.CAP_PROP_GAIN, current_gain)            # 增益
            load_eg=0
        elif key == ord('c'):  # 清除当前使用的HSV范围
            current_min = [0, 0, 0]
            current_max = [0, 0, 0]
            print("已清除当前使用的HSV范围")
        elif key == ord('q'):
            cap.set(cv2.CAP_PROP_EXPOSURE, current_exp)      # 曝光
            cap.set(cv2.CAP_PROP_GAIN, current_gain)            # 增益
            
            
        # 切换调整模式
        elif H_low_key==1: # 切换H最小值调整模式
            h_low_fix = 1 - h_low_fix
            print(f"H最小值调整模式: {'开启' if h_low_fix else '关闭'}")
            H_low_key=0
        elif H_up_key==1: # 切换H最大值调整模式
            h_up_fix = 1 - h_up_fix
            print(f"H最大值调整模式: {'开启' if h_up_fix else '关闭'}")
            H_up_key=0
        elif S_low_key==1: # 切换S最小值调整模式
            s_low_fix = 1 - s_low_fix
            print(f"S最小值调整模式: {'开启' if s_low_fix else '关闭'}")
            S_low_key=0
        elif S_up_key==1: # 切换S最大值调整模式
            s_up_fix = 1 - s_up_fix
            print(f"S最大值调整模式: {'开启' if s_up_fix else '关闭'}")
            S_up_key=0
        elif V_low_key==1: # 切换V最小值调整模式
            v_low_fix = 1 - v_low_fix
            print(f"V最小值调整模式: {'开启' if v_low_fix else '关闭'}")
            V_low_key=0
        elif V_up_key==1: # 切换V最大值调整模式
            v_up_fix = 1 - v_up_fix
            print(f"V最大值调整模式: {'开启' if v_up_fix else '关闭'}")
            V_up_key=0
        elif exp_key==1: # 切换曝光调整模式
            exp_fix = 1 - exp_fix
            print(f"曝光调整模式: {'开启' if exp_fix else '关闭'}")
            exp_key=0
        elif gain_key==1: # 切换增益调整模式
            gain_fix = 1 - gain_fix
            print(f"增益调整模式: {'开启' if gain_fix else '关闭'}")
            gain_key=0
            
        # 调整HSV值
        elif count_down==1: # 减少当前选中的HSV值
            if h_low_fix:
                current_min[0] = max(0, current_min[0]-1)
                print(f"H_min减少至: {current_min[0]}")
            elif h_up_fix:
                current_max[0] = max(0, current_max[0]-1)
                print(f"H_max减少至: {current_max[0]}")
            elif s_low_fix:
                current_min[1] = max(0, current_min[1]-1)
                print(f"S_min减少至: {current_min[1]}")
            elif s_up_fix:
                current_max[1] = max(0, current_max[1]-1)
                print(f"S_max减少至: {current_max[1]}")
            elif v_low_fix:
                current_min[2] = max(0, current_min[2]-1)
                print(f"V_min减少至: {current_min[2]}")
            elif v_up_fix:
                current_max[2] = max(0, current_max[2]-1)
                print(f"V_max减少至: {current_max[2]}")
            elif exp_fix:
                current_exp=max(0,current_exp-50)
                print(f"曝光减少至: {current_exp}")
            elif gain_fix:
                current_gain=max(0,current_gain-1)
                print(f"增益减少至: {current_gain}")
            count_down=0
        elif count_up==1: # 增加当前选中的HSV值
            if h_low_fix:
                current_min[0] = min(179, current_min[0]+1)
                print(f"H_min增加至: {current_min[0]}")
            elif h_up_fix:
                current_max[0] = min(179, current_max[0]+1)
                print(f"H_max增加至: {current_max[0]}")
            elif s_low_fix:
                current_min[1] = min(255, current_min[1]+1)
                print(f"S_min增加至: {current_min[1]}")
            elif s_up_fix:
                current_max[1] = min(255, current_max[1]+1)
                print(f"S_max增加至: {current_max[1]}")
            elif v_low_fix:
                current_min[2] = min(255, current_min[2]+1)
                print(f"V_min增加至: {current_min[2]}")
            elif v_up_fix:
                current_max[2] = min(255, current_max[2]+1)
                print(f"V_max增加至: {current_max[2]}")
            elif exp_fix:
                current_exp=max(0,current_exp+50)
                print(f"曝光增加至: {current_exp}")
            elif gain_fix:
                current_gain=max(0,current_gain+1)
                print(f"增益增加至: {current_gain}")
            count_up=0
    
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()
