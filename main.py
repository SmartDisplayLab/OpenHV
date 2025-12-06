import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from codeHV.HV import *
import os
import time
import argparse
import configparser

import pyvista as pv
from pyvista import Sphere, numpy_to_texture, global_theme

from pyvistaqt import QtInteractor

from codeHV.Capture import capture

import time



IMAGE_SIZE_480 = (480, 480)
IMAGE_SIZE_320 = (320, 320)

def get_parsers():

    parser = argparse.ArgumentParser(description="程序描述")
    parser.add_argument("--config", type=str, default='config\example.cfg', help="项目路径")
    return parser.parse_args()

def read_config(file_path):
    config = configparser.ConfigParser()
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件 {file_path} 不存在")
    
    # 读取配置文件
    config.read(file_path)
    
    # 获取所有配置信息
    config_data = {}
    
    # 遍历所有section
    for section in config.sections():
        config_data[section] = {}
        
        # 遍历section中的所有键值对
        for key, value in config[section].items():
            # 尝试自动转换类型
            if value.lower() in ('true', 'yes', 'on'):
                config_data[section][key] = True
            elif value.lower() in ('false', 'no', 'off'):
                config_data[section][key] = False
            elif value.isdigit():
                config_data[section][key] = int(value)
            else:
                try:
                    # 尝试转换为浮点数
                    config_data[section][key] = float(value)
                except ValueError:
                    # 保持为字符串
                    config_data[section][key] = value
    
    return config_data



class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.config = read_config(cfg)
        self.base_dir = self.config['paths']['base_dir']
        self.unity_path = self.config['paths']['unity_path']
        self.fig_tmp_dir = os.path.join(self.base_dir, "fig_tmp")
        self.project_path = self.config['paths']['project_path']
        
        # 初始化UI和绑定事件
        self._init_ui()
        self._bind_signals()
        
        # 初始化参数和资源
        self.result_locations = [self.base_dir+f'/fig/result{i}.png'
                                 for i in range(2 + 2 + 3 + 1 + 1 + 1)]
        self.parameters = {}
        self.maskL = self.maskR = None

        self._init_blur_and_mask()

        
        
        # 加载初始配置
        self._load_settings()
        self.load_in_parameters()
        self.present_image = {'left':None, 'right':None}

        

    def _init_ui(self):
        """初始化界面组件"""
        # 初始化图片显示尺寸
        self.LeftImage.setFixedSize(*IMAGE_SIZE_480)
        self.RightImage.setFixedSize(*IMAGE_SIZE_480)

        self.real_label.setScaledContents(True)
        self.left_eye_label.setScaledContents(True)
        self.right_eye_label.setScaledContents(True)

    def _bind_signals(self):
        """绑定信号与槽"""
        self.pushButton_left.clicked.connect(self.choose_file_left)
        self.pushButton_right.clicked.connect(self.choose_file_right)
        self.pushButton_para.clicked.connect(self.load_in_parameters)
        self.pushButton_generate.clicked.connect(self.generate_img_in_unity)
        self.pushButton_init.clicked.connect(self.load_in)
        
        # 使用循环绑定功能按钮
        for btn in [self.F1, self.F2, self.F3, self.F4, self.F5]:
            btn.clicked.connect(self.load_in)
        
        self.show_fig_in_retina_button.clicked.connect(self.show_fig_in_retina)

        self.input_axis_ocuil.clicked.connect(self.get_axis_ocuil)

        self.pushButtontry.clicked.connect(self.captureshow)
        self.pushButtontry_2.clicked.connect(self.endshow)

    
    def _load_settings(self):
        settings = self.config['settings']
        self.textEdit.setText(str(settings['focuslength']))
        self.comboBox_focus.setCurrentIndex(int(settings['focustype']))
        self.textEdit_3.setText(str(settings['fov']))
        self.textEdit_4.setText(str(settings['pupillength']))
        self.textEdit_2.setText(str(settings['position']))
        #self.parameters["farClip"] = float(lines[5][1])
        self.parameters["farClip"] = 0.0

        self.cap = None


    def _init_blur_and_mask(self):
        text1="Current half ALRR:"
        text2="Enter it below to view the retinal projection in 'In Retina' on the left."
        self.axis_ocuil = 1.5
        self.left_retina_plotter = QtInteractor(parent=self.vtkWidgetleft)
        self.right_retina_plotter = QtInteractor(parent=self.vtkWidgetright)
        layout = QtWidgets.QVBoxLayout(self.vtkWidgetleft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.left_retina_plotter)
        layout.addWidget(self.right_retina_plotter)
        self.Retina_info_1.setText(text1)
        self.Retina_info_2.setText(text2)


    def choose_file_left(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择左眼图像", "", "图像文件 (*.jpg *.png)"
        )
        if filename:
            self.LeftText.setText(filename)

    def choose_file_right(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择右眼图像", "", "图像文件 (*.jpg *.png)"
        )
        if filename:
            self.RightText.setText(filename)

    def generate_img_in_unity(self):
        '''"""调用Unity生成图像"""
        unity_exe = self.unity_path
        project_path = self.project_path
        log_path = os.path.join(project_path, "unity_log.log")

        def kill_unity():
            os.system("taskkill /IM Unity.exe /F")

        def clear_log():
            if os.path.exists(log_path):
                os.remove(log_path)

        kill_unity()
        time.sleep(1)
        clear_log()

        cmd = [
            unity_exe,
            "-quit",
            "-projectPath", project_path,
            "-logFile", log_path,
            "-executeMethod", "StaticScreenCapture.CaptureScreen"
        ]
        
        import subprocess
        subprocess.run(cmd, check=True)'''
        if self.cap is None or self.cap.latest_frames['left'] is None:
            return
        self.present_image['left'] = self.cap.latest_frames["left"]
        self.present_image['right'] = self.cap.latest_frames["right"]

        cv2.imwrite(self.result_locations[0], self.present_image['left'])
        cv2.imwrite(self.result_locations[1], self.present_image['right'])

        #f0(left, right, self.result_locations[0], self.result_locations[1])
        
        self._set_pixmap(self.LeftImage, self.present_image['left'], IMAGE_SIZE_480)
        self._set_pixmap(self.RightImage, self.present_image['right'], IMAGE_SIZE_480)

    def get_axis_ocuil(self):
        text = self.axis_ocuil_input.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "输入错误", "请输入一个数字！")
        else:       
            try:
                # 转换为数字（可改为 int(text) 如果只接受整数）
                self.axis_ocuil = float(text)
                #QtWidgets.QMessageBox.information(self, "输入有效", f"你输入的数字是：{self.axis_ocuil}")
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "类型错误", "输入的不是有效的数字，请重新输入。")
        #self.axis_ocuil_input.clear()
        self.left_retina_plotter.clear()
        self.right_retina_plotter.clear()


    def show_fig_in_retina(self):
        fig_left = cv2.imread('fig/result0.png')
        fig_right = cv2.imread('fig/result1.png')
        
        self.standalone_visualize(self.left_retina_plotter,self.vtkWidgetleft,fig_left,120,60,1860,1860)
        self.standalone_visualize(self.right_retina_plotter,self.vtkWidgetright,fig_right,120,60,1860,1860)
        #plotter.show()

    def standalone_visualize(self, plotter, vtkWidget, image, tex_h_fov, tex_v_fov, 
                         theta_res=120, phi_res=120,
                         save_path=None, show=True):
        """
        渲染球面投影到指定的 QtInteractor（plotter）中，嵌入到 vtkWidget。
        """

        if vtkWidget.layout() is None:
            layout = QtWidgets.QVBoxLayout(vtkWidget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(plotter)

        plotter.set_background("black") 

        # 生成球体网格
        sphere = pv.Sphere(theta_resolution=400, phi_resolution=400)
        ellipsoid = sphere.scale([1.0, self.axis_ocuil, 1.0], inplace=False)

        # 截取后半球：保留 y <= 0 的部分
        back_hemisphere = ellipsoid.clip(normal=(0, 1, 0), origin=(0, 0, 0))
        back_hemisphere.texture_map_to_sphere(inplace=True, prevent_seam=True)
        back_hemisphere = back_hemisphere.flip_faces()
        texture = pv.numpy_to_texture(image)
        plotter.add_mesh(back_hemisphere, texture=texture)

        # 前半球叠加
        ellipsoid2 = pv.Sphere(theta_resolution=400, phi_resolution=400)
        front_hemisphere = ellipsoid2.clip(normal=(0, -1, 0), origin=(0, 0, 0))
        front_hemisphere.texture_map_to_sphere(inplace=True, prevent_seam=True)
        front_hemisphere.flip_faces(inplace=True)
        eye_fig = cv2.imread("fig/eye_diagram.png")
        texture2 = pv.numpy_to_texture(eye_fig)
        plotter.add_mesh(front_hemisphere, texture=texture2)

        # 设置摄像机参数（不调用 reset_camera() 避免打断用户交互）
        if show:
            plotter.camera.position = (0, 0.1, 0)        # 比原点稍远
            plotter.camera.focal_point = (0, 0, 0)       # 观察球心
            plotter.camera.up = (0, 0, 1)
            
            plotter.set_focus((0, 0, 0))                 # 设置旋转中心为球心

    
    def load_in_parameters(self):
        """加载处理参数"""
        para_map = [
            ("FocusLength", self.textEdit, float),
            ("FocusType", self.comboBox_focus, lambda x: 0 if x == "Finite" else 1),
            ("FOV", self.textEdit_3, float),
            ("pupilLength", self.textEdit_4, float),
            ("position", self.textEdit_2, int),
        ]

        # 更新参数
        for name, widget, converter in para_map:
            if isinstance(widget, QtWidgets.QComboBox):
                value = converter(widget.currentText())
            else:
                value = converter(widget.toPlainText())
            self.parameters[name] = value

        # 处理无限对焦
        if self.parameters["FocusType"] == 1:
            self.parameters["FocusLength"] = 1e5

        # 生成掩模
        self._generate_masks()

        

    def _generate_masks(self):
        """生成视觉掩模"""
        mask_r = cv2.imread(r"fig/NEWmask_r_164.png")
        mask_l = np.flip(mask_r, axis=1)
        h, w, _ = mask_r.shape

        # 计算缩放比例
        fov_rad = np.radians(self.parameters["FOV"])
        scale = np.tan(fov_rad) / np.tan(np.radians(164))
        new_size = (int(w * scale), int(h * scale))

        # 裁剪中心区域
        def center_crop(img, size):
            dh = (h - size[1]) // 2
            dw = (w - size[0]) // 2
            return img[dh:dh+size[1], dw:dw+size[0]]

        mask_r = center_crop(mask_r, new_size)
        mask_l = center_crop(mask_l, new_size)

        # 调整尺寸并添加盲区
        target_size = (1860, 1860)
        self.maskL_noBlind = cv2.resize(mask_l, target_size)
        self.maskR_noBlind = cv2.resize(mask_r, target_size)
        self.maskL = add_blind(copy.deepcopy(self.maskL_noBlind), "left")
        self.maskR = add_blind(copy.deepcopy(self.maskR_noBlind), "right")

        #self.maskL = cv2.resize(mask_l, target_size)
        #self.maskR = cv2.resize(mask_r, target_size)

    def load_in(self):
        """根据当前选中的列表项加载对应内容"""
        func_handlers = {
            0: self._handle_raw_images,
            1: self._handle_blurred_images,
            2: self._handle_binocular_fusion,
            3: self._handle_depth_map,
            4: self._handle_edge_detection,
            5: self._handle_saliency_detection,
        }

        # 使用 QListWidget 获取当前行号
        current_index = self.listWidget.currentRow()  # 👈 listWidget 是 QListWidget 的对象名
        handler = func_handlers.get(current_index)
        if handler:
            handler()

    def _handle_raw_images(self):
        """处理原始图像显示"""
        left = self.LeftText.toPlainText()
        right = self.RightText.toPlainText()
        print(left,right)
        self.present_image['left'], self.present_image['right'] = f0(left, right, self.result_locations[0], self.result_locations[1])
        
        self._set_pixmap(self.LeftImage, self.present_image['left'], IMAGE_SIZE_480)
        self._set_pixmap(self.RightImage, self.present_image['right'], IMAGE_SIZE_480)

    def _handle_blurred_images(self):
        """处理模糊图像显示"""
        
        left_blured_img, right_blured_img = blur(self.present_image, self.result_locations[2], 
                                                 self.result_locations[3], self.maskL, self.maskR)
        
        
        
        self._set_pixmap(self.ImageF1_L, left_blured_img)
        self._set_pixmap(self.ImageF1_R, right_blured_img)

        
    def _handle_binocular_fusion(self):
        """处理双目融合"""
        
        params = [
            self.parameters["FOV"], 
            self.parameters["pupilLength"],
            self.parameters["FocusLength"],
            self.maskL,
            self.maskR
        ]
        fusioned, fusioned_l, fusioned_r = binocular_fusion(self.present_image, self.result_locations[4], 
                                                            self.result_locations[5],self.result_locations[6], *params)
        
        self._set_pixmap(self.ImageF2, fusioned, IMAGE_SIZE_320)
        self._set_pixmap(self.ImageF2_L, fusioned_l, IMAGE_SIZE_320)
        self._set_pixmap(self.ImageF2_R, fusioned_r, IMAGE_SIZE_320)

    def _handle_depth_map(self):
        """处理深度图"""
        params = [
            self.parameters["FOV"],
            self.parameters["pupilLength"],
            self.parameters["FocusLength"],
            self.maskL,
            self.maskR
        ]
        depth_map = compute_depth_map(self.present_image, self.result_locations[7], *params)
        self._set_pixmap(self.ImageF3, depth_map)

    def _handle_edge_detection(self):
        """处理边缘检测"""
        result = edge_detection(self.result_locations[4], self.result_locations[8])
        self._set_pixmap(self.ImageF4_L, result)

    def _handle_saliency_detection(self):
        """处理显著性检测"""
        result = segment_saliency(self.result_locations[4], self.result_locations[9])
        self._set_pixmap(self.ImageF5_L, result)

    def _set_pixmap(self, widget, img: np.ndarray, size=None):
        """通用设置图片方法（支持numpy图像）"""

        # 检查维度并转换成RGB格式
        if img.ndim == 2:
            # 灰度图
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        elif img.ndim == 3:
            if img.shape[2] == 3:
                # OpenCV默认是BGR，需要转为RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format_RGB888)
            elif img.shape[2] == 4:
                # 带alpha通道
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGBA8888)
            else:
                raise ValueError("Unsupported image channel number.")
        else:
            raise ValueError("Unsupported image shape.")

        # 转为QPixmap
        pixmap = QPixmap.fromImage(qimg)

        # 按需缩放
        if size:
            pixmap = pixmap.scaled(*size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        widget.setPixmap(pixmap)

    def captureshow(self):
        import threading
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QImage, QPixmap

        host = '127.0.0.1'
        main_port = 5001
        left_port = 5002
        right_port = 5003

        self.cap = capture(host)
        threading.Thread(target=self.cap.receive_stream, args=("main", main_port), daemon=True).start()
        threading.Thread(target=self.cap.receive_stream, args=("left", left_port), daemon=True).start()
        threading.Thread(target=self.cap.receive_stream, args=("right", right_port), daemon=True).start()

        import cProfile
        import pstats
        import io

        if self.maskL is None:
                self._generate_masks()


        def cvimg_to_qpixmap(cv_img):
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
            q_img._ref = cv_img  # 防止被 GC
            return QPixmap.fromImage(q_img)

        def update_label_image(name, label):

            

            #print(self.maskL.shape) (1860,1860,3)
            h, w, _ = self.maskL.shape

            frame = self.cap.latest_frames.get(name)
            '''if frame is None and name != "fusion":
                return'''

            if name == "fusion":
                frame = np.zeros((h,w))
            
            if frame is None:
                return
            
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            if name == "left":
                frame = add_mask(frame, self.maskL)
            if name == "right":
                frame = add_mask(frame, self.maskR)

            if name == "fusion":
                left_frame = self.cap.latest_frames.get("left")
                right_frame = self.cap.latest_frames.get("right")
                
                if left_frame is None:
                    return
                
                left_frame = cv2.resize(left_frame,(h,w))
                right_frame = cv2.resize(right_frame,(h,w))

                params = [
                    self.parameters["FOV"], 
                    self.parameters["pupilLength"],
                    self.parameters["FocusLength"],
                    self.maskL,
                    self.maskR
                ]

                profiler = cProfile.Profile()
                profiler.enable()
                frame = binocular_fusion(left_frame, right_frame, self.result_locations[4], self.result_locations[5],
                        self.result_locations[6], *params)
                profiler.disable()

                stats = pstats.Stats(profiler).sort_stats('cumtime')
                stats.print_stats(10)  # 显示前10条最耗时函数
                
                if frame is None:
                    return

            
            pixmap = cvimg_to_qpixmap(frame)
            '''if pixmap:
                label.setPixmap(pixmap.scaled(
                    label.width(), label.height(), Qt.KeepAspectRatio))'''
            if pixmap:
                label.setPixmap(pixmap)

        def refresh_labels():
            
            start = time.perf_counter()
            update_label_image("main", self.real_label)
            update_label_image("left", self.left_eye_label)
            update_label_image("right", self.right_eye_label)
            #update_label_image("fusion", self.fusion_eye_label)
            end = time.perf_counter()  
            delay = end - start
            #print(f"执行延迟: {delay*1000:.3f} ms")

        # 👇 一定要用 self.timer，而不是局部变量
        self.timer = QTimer()
        self.timer.timeout.connect(lambda: refresh_labels())
        self.timer.start(33)


    def endshow(self):
        """结束图像显示并清空 QLabel"""
        # 1️⃣ 停止定时器
        if getattr(self, "timer", None) is not None and self.timer.isActive():
            self.timer.stop()
            self.timer = None

        # 2️⃣ 关闭接收线程（可选）
        if hasattr(self, "cap"):
            try:
                # 如果你在 capture 类里加了 stop 标志，可以在这里设置：
                self.cap.running = False
                self.cap = None
            except Exception as e:
                print(f"Error closing capture: {e}")

        # 3️⃣ 清空 QLabel 内容
        if hasattr(self, "real_label"):
            self.real_label.clear()
        if hasattr(self, "left_eye_label"):
            self.left_eye_label.clear()
        if hasattr(self, "right_eye_label"):
            self.right_eye_label.clear()
        if hasattr(self, "fusion_eye_label"):
            self.fusion_eye_label.clear()

        print("显示已结束，QLabel 已清空。")

    def closeEvent(self, event):
        # 安全销毁两个 plotter
        try:
            self.left_retina_plotter.close()
            self.left_retina_plotter.interactor.close()
        except Exception as e:
            print("[DEBUG] Failed to close left plotter:", e)

        try:
            self.right_retina_plotter.close()
            self.right_retina_plotter.interactor.close()
        except Exception as e:
            print("[DEBUG] Failed to close right plotter:", e)

        event.accept()


if __name__ == "__main__":
    args=get_parsers()
    config_file = args.config
    app = QApplication(sys.argv)
    window = MyWindow(config_file)
    window.show()
    sys.exit(app.exec_())