import cv2


def Visual_Saliency_Detection(image):
    # 初始化OpenCV的静态显著性检测器
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    # 计算图像的显著性区域
    (success, saliencyMap) = saliency.computeSaliency(image)
    # 将显著性区域转换为8位无符号整数

    saliencyMap = (saliencyMap * 255).astype("uint8")
    return saliencyMap
