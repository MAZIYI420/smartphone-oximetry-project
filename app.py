# 导入所有必需的库
import streamlit as st
import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- [第 1 部分：从 oximetry_analysis_dl.py 复制过来的辅助函数] ---
# 我们需要重新定义这些函数，因为这个 app 是一个独立的文件

MODEL_FILENAME = 'spo2_model_v8_final.h5'
FRAME_RATE = 30  # 摄像头的帧率
DURATION = 3     # 3 秒
FRAME_COUNT = FRAME_RATE * DURATION  # 90 帧

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    设计一个带通滤波器。
    在 oximetry_analysis_dl.py 中, 我们使用了 0.5 Hz 到 4 Hz。
    (0.5 Hz * 60 = 30 bpm; 4 Hz * 60 = 240 bpm)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """应用带通滤波器"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def preprocess_signals(raw_signals, fs=FRAME_RATE):
    """
    对原始 R, G, B 信号进行滤波和标准化。
    输入形态: (90, 3)
    """
    filtered_signals = np.zeros_like(raw_signals)
    
    # 1. 滤波
    lowcut = 0.5  # 对应 30 bpm
    highcut = 4.0   # 对应 240 bpm
    for i in range(raw_signals.shape[1]): # 对 R, G, B 三个通道分别滤波
        filtered_signals[:, i] = butter_bandpass_filter(raw_signals[:, i], lowcut, highcut, fs)
        
    # 2. 标准化 (Z-score)
    # 这是训练 CNN 模型的标准做法
    mean = np.mean(filtered_signals, axis=0)
    std = np.std(filtered_signals, axis=0)
    
    # 防止除以零
    std[std == 0] = 1e-10
    
    standardized_signals = (filtered_signals - mean) / std
    
    return standardized_signals, filtered_signals # 返回两种信号用于绘图

# --- [第 2 部分：Streamlit 应用界面] ---

st.set_page_config(page_title="实时血氧检测仪", page_icon="❤️")
st.title("❤️ 实时血氧检测仪 (Beta)")
st.write("---")

# 检查模型文件是否存在
if not os.path.exists(MODEL_FILENAME):
    st.error(f"**错误：** 找不到模型文件 `{MODEL_FILENAME}`。")
    st.write("请确保您已经成功运行了 `oximetry_analysis_dl.py` 来训练并生成模型文件，并将其放在与此 `app.py` 相同的文件夹中。")
else:
    # 加载模型 (使用 @st.cache_resource 来避免每次都重新加载)
    @st.cache_resource
    def load_spo2_model():
        try:
            # 使用 TensorFlow 的 Keras API 加载模型
            model = tf.keras.models.load_model(MODEL_FILENAME)
            return model
        except Exception as e:
            st.error(f"加载模型时出错: {e}")
            return None

    model = load_spo2_model()
    
    if model:
        st.success(f"成功加载AI模型: `{MODEL_FILENAME}`")
        st.info("请将您的**食指**完全覆盖在电脑的**网络摄像头**上，然后点击“开始测量”。")

        # 使用 session_state 来管理测量状态
        if 'measuring' not in st.session_state:
            st.session_state.measuring = False
        if 'results' not in st.session_state:
            st.session_state.results = None

        if st.button("开始测量", type="primary", disabled=st.session_state.measuring):
            st.session_state.measuring = True
            st.session_state.results = None
            
            # --- 实时视频采集 ---
            try:
                cap = cv2.VideoCapture(0) # 0 代表默认的摄像头
                
                if not cap.isOpened():
                    st.error("无法打开摄像头。请检查摄像头权限或连接。")
                    st.session_state.measuring = False
                else:
                    raw_signals_list = []
                    
                    # 创建一个用于显示倒计时和状态的占位符
                    status_placeholder = st.empty()
                    
                    # 倒计时
                    for i in range(3, 0, -1):
                        status_placeholder.warning(f"请保持手指稳定... **{i}**")
                        time.sleep(1)
                    
                    status_placeholder.info(f"正在采集中... (共 {DURATION} 秒)")
                    
                    start_time = time.time()
                    
                    while len(raw_signals_list) < FRAME_COUNT:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("读取帧失败，停止测量。")
                            break
                        
                        # (关键步骤) 提取 R, G, B 平均值
                        # 我们不再保存视频，而是实时计算
                        # 我们假设手指覆盖了画面的中心区域
                        
                        h, w, _ = frame.shape
                        center_x, center_y = w // 2, h // 2
                        box_size = min(w, h) // 4  # 取中心 1/4 的区域
                        
                        roi = frame[center_y - box_size : center_y + box_size,
                                  center_x - box_size : center_x + box_size]
                        
                        if roi.size == 0:
                            continue # 以防万一 ROI 无效
                        
                        # 计算 R, G, B 平均值
                        # 注意: OpenCV 的顺序是 BGR
                        avg_b = np.mean(roi[:, :, 0])
                        avg_g = np.mean(roi[:, :, 1])
                        avg_r = np.mean(roi[:, :, 2])
                        
                        raw_signals_list.append([avg_r, avg_g, avg_b])
                        
                        # 简单的帧率控制 (非精确，但足够用)
                        time.sleep(1.0 / FRAME_RATE) 
                    
                    # 采集完毕，释放摄像头
                    cap.release()
                    
                    if len(raw_signals_list) == FRAME_COUNT:
                        status_placeholder.success("采集完成！正在分析...")
                        
                        raw_signals_np = np.array(raw_signals_list)
                        
                        # --- 模型预测 ---
                        # 1. 预处理
                        preprocessed_data, filtered_data_for_plot = preprocess_signals(raw_signals_np)
                        
                        # 2. 准备模型输入 (模型期望的输入是 (1, 90, 3))
                        model_input = np.expand_dims(preprocessed_data, axis=0)
                        
                        # 3. 预测
                        prediction = model.predict(model_input)
                        predicted_spo2 = prediction[0][0]
                        
                        # 储存结果
                        st.session_state.results = {
                            "spo2": predicted_spo2,
                            "filtered": filtered_data_for_plot,
                            "raw": raw_signals_np
                        }
                        
                    else:
                        status_placeholder.error("未能采集到足够的帧，请重试。")

            except Exception as e:
                st.error(f"测量过程中发生错误: {e}")
            
            # 测量结束
            st.session_state.measuring = False
            st.rerun() # 重新运行脚本以显示结果

        # --- 显示结果 ---
        if st.session_state.results:
            st.write("---")
            st.header("测量结果")
            
            spo2_val = st.session_state.results["spo2"]
            
            # 根据血氧值显示不同的颜色
            if spo2_val >= 94:
                st.metric(label="预测血氧饱和度 (SpO2)", value=f"{spo2_val:.2f} %", delta="状态良好")
            elif 90 <= spo2_val < 94:
                st.metric(label="预测血氧饱和度 (SpO2)", value=f"{spo2_val:.2f} %", delta="轻度缺氧 - 请注意", delta_color="inverse")
            else:
                st.metric(label="预测血氧饱和度 (SpO2)", value=f"{spo2_val:.2f} %", delta="显著缺氧 - 警告", delta_color="inverse")

            st.info("免责声明：本结果由AI模型预测，仅供参考，不能替代专业医疗诊断。")

            # --- 绘制图表 ---
            st.subheader("信号分析图")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            time_axis = np.arange(FRAME_COUNT) / FRAME_RATE
            
            # 图 1: 原始信号
            ax1.set_title("1. 摄像头捕获的原始信号 (Raw)")
            ax1.plot(time_axis, st.session_state.results["raw"][:, 0], 'r-', label="Red (原始)")
            ax1.plot(time_axis, st.session_state.results["raw"][:, 1], 'g-', label="Green (原始)")
            ax1.plot(time_axis, st.session_state.results["raw"][:, 2], 'b-', label="Blue (原始)")
            ax1.set_ylabel("像素均值")
            ax1.legend()
            
            # 图 2: 滤波后的信号 (用于预测)
            ax2.set_title("2. 滤波后的脉搏信号 (Filtered)")
            ax2.plot(time_axis, st.session_state.results["filtered"][:, 0], 'r-', label="Red (滤波后)")
            ax2.plot(time_axis, st.session_state.results["filtered"][:, 1], 'g-', label="Green (滤波后)")
            ax2.plot(time_axis, st.session_state.results["filtered"][:, 2], 'b-', label="Blue (滤波后)")
            ax2.set_xlabel("时间 (秒)")
            ax2.set_ylabel("标准化信号")
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)

