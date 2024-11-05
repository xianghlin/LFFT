import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utilss import GradCAM, show_cam_on_image, center_crop_img
import torch
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import cairosvg

class ReshapeTransform:
    def __init__(self, model):
        input_size = 256
        patch_size = 16
        self.h = 16  #计算垂直方向切片数量
        self.w = 16  #计算呢水平方向切片数量

    def __call__(self, x):
        # 移除CLS令牌并重塑张量
        # [batch_size, num_tokens, token_dim]
        # result = x[:, 1:, :].reshape(x.size(0),
        #                              16,
        #                              16,
        #                              768)
        result = x.reshape(x.size(0),
                           16,
                           16,
                           768)

        # 将通道维度移到第一维，类似于CNN
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    # model = vit_base_patch16_224()
    # # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    # weights_path = "./vit_base_patch16_224.pth"
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    # 选择用于生成CAM的目标层
    # 由于最后的分类是在最后一个注意力块中计算的，所以输出不受最后一层的影响
    # 最后一层的输出梯度为0
    # 我们应该选择最后一个注意力块之前的任何一层

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/VeRi/vit_base.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    print(model)

    target_layers = [model.b1_block.norm1]
    # target_layers = [model.b1[0].norm1]

    # 数据变换
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = "picture/0210_c009_00072790_0.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 256)
    # plt.imshow(img)
    # plt.axis('off')  # 去掉坐标轴
    # plt.savefig("picture/477.svg", format="svg", bbox_inches="tight")
    # plt.show()


    # [C, H, W]
    img_tensor = data_transform(img)
    # 维度批次扩展
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 创建GradCAM对象
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = None # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig("picture/210oh.svg", format="svg", bbox_inches="tight")
    plt.show()


