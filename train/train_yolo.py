from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-seg.pt")  # load a pretrained model (recommended for training)

# 训练模型
model.train(
    data="/data/lijun/model/court/train/train.yaml",  # 数据集配置文件路径
    epochs=50,  # 训练轮数
    imgsz=320,  # 输入图像大小
    device=0,  # 使用的设备（GPU 3）
    batch=16,  # 每批次的图像数量

    # 数据增强超参数
    hsv_h=0.01,  # HSV-Hue 增强
    hsv_s=0.7,  # HSV-Saturation 增强
    hsv_v=0.4,  # HSV-Value 增强
    flipud=0.5,  # 上下翻转概率
    fliplr=0.5,  # 左右翻转概率
    mosaic=0.0,  # Mosaic 数据增强概率
    mixup=0.0,  # MixUp 数据增强概率
    degrees=45.0,  # 随机旋转角度范围
    translate=0.1,  # 随机平移范围
    scale=0.3,  # 随机缩放范围
    shear=45.0,  # 随机剪切角度范围
    # copy_paste= 0.1,  # 复制粘贴增强概率
)