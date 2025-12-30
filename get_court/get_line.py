import sys
sys.path.append('/data/lijun/model/court/get_court/RS')

import cv2
import os
import numpy as np
from ultralytics import YOLO
from RS import inference_invsr 

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class LineSegmentor:
    def __init__(self, model_path, conf=0.3):
        """
        初始化分割器
        :param model_path: YOLO模型路径
        :param input_folder: 输入图片文件夹
        :param output_folder: 输出结果文件夹
        :param conf: 置信度阈值
        :param classes: 指定检测类别
        """
        # 初始化模型
        self.model = YOLO(model_path,verbose=False)
        self.conf = conf
        self.classes = [0]  # 仅检测类别0（假设为网球场线）
        
        # 可视化参数
        self.mask_alpha = 0.8
        self.box_color = (0, 255, 0)  # BGR格式
        
        # 验证模型类型
        if self.model.task != 'segment':
            raise ValueError("model type must be 'segment' for segmentation tasks.")

    def process_per_image(self, img):
        """
        处理单张图片并保存结果
        :param img: 输入图片
        :param img_name: 图片名称
        :return: None
        """
        # 转换颜色空间
        img_rgb = img

        # 执行预测
        results = self.model.predict(
            img_rgb,
            conf=self.conf,
            classes=self.classes,
            imgsz=320
        )

        # 处理分割结果
        self.overlay = img.copy()
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                # 合并所有掩码
                for mask in result.masks:
                    mask_data = mask.data[0].cpu().numpy()
                    mask_uint8 = (mask_data * 255).astype(np.uint8)
                    mask_uint8 = cv2.resize(mask_uint8, (img.shape[1], img.shape[0]))
                    _, binary_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
                    combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
        
        # combined_mask 上下左右翻转
        self.output_mask = cv2.flip(combined_mask, -1)
        return self.output_mask

    def vis(self, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 读取输入图像
        input_img = cv2.imread(input_path)
        
        # 将mask resize到原图大小
        mask_resized = cv2.resize(self.output_mask, (input_img.shape[1], input_img.shape[0]))
        
        # 可视化所有mask和轮廓
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(input_img, [contour], -1, self.box_color, -1)
            cv2.drawContours(input_img, contours, -1, (255, 0, 0), 1)
        
        # 融合原图和掩码
        blended = cv2.addWeighted(input_img, 1 - self.mask_alpha, input_img, self.mask_alpha, 0)
        
        # 保存结果图像
        save_img_path = os.path.join(output_path, os.path.basename(input_path))
        
        # 原图和blended图拼接保存
        concat_img = np.concatenate((cv2.imread(input_path), blended), axis=1)
        cv2.imwrite(save_img_path, concat_img)
        

if __name__ == "__main__":
    # 初始化网球场地分割模型
    segmentor = LineSegmentor(
        model_path="/data/lijun/model/ultralytics/runs/segment/train11/weights/best.pt",
        conf=0.35
    )
    
    # 初始化InvSR模型
    args = inference_invsr.get_parser()

    configs = inference_invsr.get_configs(args)

    sampler = inference_invsr.InvSamplerSR(configs)
    
    # 所有待处理图像所在文件夹
    img_fold = '/data/lijun/model/court/imgs_12_26/tmp'
    output_fold = '/data/lijun/model/court/imgs_12_26/tmp_out'
    
    # 遍历文件夹中的所有图片
    all_imgs = os.listdir(img_fold)
    
    for img_name in all_imgs:
        print(f"Processing image: {img_name}")
        img_path = os.path.join(img_fold, img_name)
        
        # 对图像进行超分
        sr_img_bgr = sampler.inference(img_path, bs=args.bs)
        
        # 对超分后的图进行保存
        cv2.imwrite("/data/lijun/model/court/77322_221_1761221774503_ball_line_sr.png", sr_img_bgr)

        # 对超分后的图像进行分割
        segmentor.process_per_image(sr_img_bgr)
        segmentor.vis(img_path, output_fold)

        pass
    

