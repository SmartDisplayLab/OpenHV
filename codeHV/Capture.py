import socket
import struct
import threading
import cv2
import numpy as np
import win32gui, win32con
from collections import defaultdict

class capture():

    def __init__(self, host) -> None:
        self.host = host
        #self.latest_frames = {"left": None, "right": None}
        self.latest_frames = defaultdict(lambda: None)
        self.running = True

    def receive_stream(self, name, port):
        """从指定端口接收图像流"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, port))
        s.listen(1)
        print(f"[{name}] Waiting for Unity connection on port {port}...")
        conn, addr = s.accept()
        print(f"[{name}] Connected by {addr}")

        while self.running:
            try:
                # 读取帧长度（4字节，小端序）
                length_bytes = conn.recv(4)
                if not length_bytes:
                    break
                length = struct.unpack('<I', length_bytes)[0]

                # 接收完整数据
                data = b''
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet:
                        break
                    data += packet

                # 解码为图像
                img_array = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.latest_frames[name] = frame
            except Exception as e:
                print(f"[{name}] Connection error: {e}")
                break

        conn.close()
        s.close()
        print(f"[{name}] Closed connection.")

    def get_last_frame(self):
        return self.latest_frames
    


    def show_stereo_window(self):
        """同时显示左右眼拼接的画面"""
        window_name = "Stereo View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 1000, 100)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # === 在循环开始前设置窗口属性 ===
        
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 让窗口保持置顶
        cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, 0) # 可调整大小

        hwnd = win32gui.FindWindow(None, window_name)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 1000, 100, 640, 360, 0)

        while True:
            left = self.latest_frames["left"]
            right = self.latest_frames["right"]

            if left is not None and right is not None:
                # 调整尺寸保持一致
                h = min(left.shape[0], right.shape[0])
                w = min(left.shape[1], right.shape[1])
                left_resized = cv2.resize(left, (w, h))
                right_resized = cv2.resize(right, (w, h))

                # 水平拼接
                stereo = np.hstack((left_resized, right_resized))
                cv2.imshow("Stereo View", stereo)

            if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
                break

        cv2.destroyAllWindows()

def main():

    HOST = '127.0.0.1'
    LEFT_PORT = 5002
    RIGHT_PORT = 5003

    cap=capture(HOST)

    threading.Thread(target=cap.receive_stream, args=("left", LEFT_PORT), daemon=True).start()
    threading.Thread(target=cap.receive_stream, args=("right", RIGHT_PORT), daemon=True).start()

    cap.show_stereo_window()

if __name__ == "__main__":
    main()
