import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from HV import *
import os
import time
import argparse

import pyvista as pv
from pyvista import Sphere, numpy_to_texture, global_theme

from pyvistaqt import QtInteractor


# 常量定义
#BASE_DIR = r"D:\CODES\master\HumanVisionSimulatorUI"
#UNITY_PATH=r"G:\Program Files\Unity\Hub\Editor\2022.3.55f1c1\Editor\Unity.exe"
#UNITY_SETTINGS_PATH = r"G:\Program Files\Unity\projects\HDRP\Assets\Resources\setting.txt"
IMAGE_SIZE_480 = (480, 480)
IMAGE_SIZE_320 = (320, 320)
#RESULT_BASE = os.path.join(BASE_DIR, "fig", "result{}.png")
#FIG_TMP_DIR = os.path.join(BASE_DIR, "fig_tmp")

def get_parsers():

    parser = argparse.ArgumentParser(description="程序描述")
    parser.add_argument("--base_dir", type=str, default=r"D:\CODES\master\HumanVisionSimulatorUI", help="项目路径")
    parser.add_argument("--unity_path", type=str, default=r"G:\Program Files\Unity\Hub\Editor\2022.3.55f1c1\Editor\Unity.exe", help="Unity路径")
    parser.add_argument("--project_path", type=str, default=r"G:\Program Files\Unity\projects\HDRP", help="项目路径")

    return parser.parse_args()



class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.args = args
        self.base_dir = args.base_dir
        self.unity_path = args.unity_path
        #self.result_base = os.path.join(args.base_dir, "fig",f"result{i}.png")
        self.fig_tmp_dir = os.path.join(args.base_dir, "fig_tmp")
        self.project_path = args.project_path
        self.unity_settings_path = self.project_path+r"\Assets\Resources\setting.txt"
        
        # 初始化UI和绑定事件
        self._init_ui()
        self._bind_signals()
        
        # 初始化参数和资源
        #self.result_locations = [self.result_base.format(i) for i in range(2 + 2 + 3 + 1 + 1 + 1)]
        self.result_locations = [args.base_dir+f'/fig/result{i}.png'
                                 for i in range(2 + 2 + 3 + 1 + 1 + 1)]
        self.parameters = {}
        self.maskL = self.maskR = None

        self._init_blur_and_mask()

        
        
        # 加载初始配置
        self._load_settings()
        self.load_in_parameters()

        

    def _init_ui(self):
        """初始化界面组件"""
        # 初始化图片显示尺寸
        self.LeftImage.setFixedSize(*IMAGE_SIZE_480)
        self.RightImage.setFixedSize(*IMAGE_SIZE_480)

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

    def _load_settings(self):
        """加载配置文件"""
        with open(self.unity_settings_path) as f:
            lines = [line.strip().split(",") for line in f.readlines()]
            
        #print(lines)
        if len(lines) > 0:
            self.textEdit.setText(lines[0][1])
            self.comboBox_focus.setCurrentIndex(int(lines[1][1]))
            self.textEdit_3.setText(lines[2][1])
            self.textEdit_4.setText(lines[3][1])
            self.textEdit_2.setText(lines[4][1])
            #self.parameters["farClip"] = float(lines[5][1])
            self.parameters["farClip"] = 0.0

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
        """调用Unity生成图像"""
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
        subprocess.run(cmd, check=True)

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
        #print(np.shape(fig_left))

        #print(f"[DEBUG] Left widget size: {self.vtkWidgetleft.width()}x{self.vtkWidgetleft.height()}")
        #print(f"[DEBUG] Right widget size: {self.vtkWidgetright.width()}x{self.vtkWidgetright.height()}")

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
        with open(self.unity_settings_path, "w") as f:
            for name, widget, converter in para_map:
                if isinstance(widget, QtWidgets.QComboBox):
                    value = converter(widget.currentText())
                else:
                    value = converter(widget.toPlainText())
                self.parameters[name] = value
                f.write(f"{name},{value}\n")

        # 处理无限对焦
        if self.parameters["FocusType"] == 1:
            self.parameters["FocusLength"] = 1e5

        # 生成掩模
        self._generate_masks()

        # 更新文件路径
        params = self.parameters
        base_name = f"FOV{int(params['FOV'])}-F{params['FocusLength']:g}-pos{int(params['position'])}"
        self.LeftText.setText(os.path.join(self.fig_tmp_dir, f"left_{base_name}.jpg"))
        self.RightText.setText(os.path.join(self.fig_tmp_dir, f"right_{base_name}.jpg"))

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
        f0(left, right, self.result_locations[0], self.result_locations[1])
        
        self._set_pixmap(self.LeftImage, left, IMAGE_SIZE_480)
        self._set_pixmap(self.RightImage, right, IMAGE_SIZE_480)

    def _handle_blurred_images(self):
        """处理模糊图像显示"""
        left = self.LeftText.toPlainText()
        right = self.RightText.toPlainText()
        blur(left, right, self.result_locations[2], self.result_locations[3], 
           self.maskL, self.maskR)
        
        self._set_pixmap(self.ImageF1_L, self.result_locations[2])
        self._set_pixmap(self.ImageF1_R, self.result_locations[3])

        '''retina_img=cv2.imread('fig/retina.png')
        retina_rgb = cv2.cvtColor(retina_img, cv2.COLOR_BGR2RGB)

        #self._set_pixmap(self.Image_retina, retina_rgb)
        self._set_pixmap(self.Image_retina, 'fig/retina.png')'''

    def _handle_binocular_fusion(self):
        """处理双目融合"""
        left = self.LeftText.toPlainText()
        right = self.RightText.toPlainText()
        params = [
            self.parameters["FOV"], 
            self.parameters["pupilLength"],
            self.parameters["FocusLength"],
            self.maskL,
            self.maskR
        ]
        binocular_fusion(left, right, self.result_locations[4], self.result_locations[5],
           self.result_locations[6], *params)
        
        self._set_pixmap(self.ImageF2, self.result_locations[4], IMAGE_SIZE_320)
        self._set_pixmap(self.ImageF2_L, self.result_locations[5], IMAGE_SIZE_320)
        self._set_pixmap(self.ImageF2_R, self.result_locations[6], IMAGE_SIZE_320)

    def _handle_depth_map(self):
        """处理深度图"""
        left = self.LeftText.toPlainText()
        right = self.RightText.toPlainText()
        params = [
            self.parameters["FOV"],
            self.parameters["pupilLength"],
            self.parameters["FocusLength"],
            self.maskL,
            self.maskR
        ]
        compute_depth_map(left, right, self.result_locations[7], *params)
        self._set_pixmap(self.ImageF3, self.result_locations[7])

    def _handle_edge_detection(self):
        """处理边缘检测"""
        edge_detection(self.result_locations[4], self.result_locations[8])
        self._set_pixmap(self.ImageF4_L, self.result_locations[8])

    def _handle_saliency_detection(self):
        """处理显著性检测"""
        segment_saliency(self.result_locations[4], self.result_locations[9])
        self._set_pixmap(self.ImageF5_L, self.result_locations[9])

    def _set_pixmap(self, widget, path, size=None):
        """通用设置图片方法"""
        pixmap = QPixmap(path)
        if size:
            pixmap = pixmap.scaled(*size, Qt.KeepAspectRatio)
        widget.setPixmap(pixmap)

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
    app = QApplication(sys.argv)
    window = MyWindow(args)
    window.show()
    sys.exit(app.exec_())