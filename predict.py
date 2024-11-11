import os
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms

from dataset import MyData
from models.birefnet import BiRefNet
from utils import save_tensor_img, check_state_dict, get_img_files, path_to_image
from config import Config


config = Config()


transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
][False or False:])

def inference(model, image_list, pred_root, device=0):
    os.makedirs(pred_root, exist_ok=True)
    model_training = model.training
    if model_training:
        model.eval()
    for image_path in tqdm(image_list):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        image = path_to_image(image_path)
        image = transform_image(image)
        
        inputs = image.to(device).unsqueeze(0)
        with torch.no_grad():
            pred = model(inputs)[-1].sigmoid()
        print("model output: ", pred.shape)
        pred = torch.nn.functional.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        
        name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(pred_root, name+'.png')
        save_tensor_img(pred, save_path)
        
    return None


def main(args):
    # Init model

    device = config.device
    print('Testing with model {}'.format(args.ckpt))
    weights = args.ckpt
    
    image_list = sorted(get_img_files(args.img_root))

    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=False)
        
        
    print('\tInferencing {}...'.format(weights))
    # model.load_state_dict(torch.load(weights, map_location='cpu'))
    state_dict = torch.load(weights, map_location='cpu')
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    inference(
        model, image_list, pred_root=args.output,
        device=config.device
    )


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ckpt', type=str, help='model folder')
    parser.add_argument('--ckpt_folder', default=None, type=str, help='model folder')
    parser.add_argument('--output', default='output', type=str, help='Output folder')
    parser.add_argument('--img_root', type=str,help='The path of testing images')

    args = parser.parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')
    main(args)
