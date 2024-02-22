import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from glob import glob
import imageio
import torch.backends.cudnn as cudnn
from modules.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from modules.resnet import resnet50, resnet101, resnet18
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import cv2
from imagenet_index import index2class
from LRP_util import *
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--models', type=str, default='resnet50',
                    help='resnet50')
parser.add_argument('--target_layer', type=str, default='layer4',
                    help='target_layer')
parser.add_argument('--target_class', type=int, default=None,
                    help='target_class')
args = parser.parse_args()

# define data loader

###########################################################################################################################
model_arch = args.models

if model_arch == 'resnet50':
    model = resnet50(pretrained=True) #####
    if args.target_layer == 'layer1':
        target_layer = model.layer1
    elif args.target_layer == 'layer2':
        target_layer = model.layer2
    elif args.target_layer == 'layer3':
        target_layer = model.layer3
    elif args.target_layer == 'layer4':
        target_layer = model.layer4

if torch.cuda.is_available():
    model = model.cuda()
model.eval()
#######################################################################################################################

value = dict()
def forward_hook(module, input, output):
    value['activations'] = output
def backward_hook(module, input, output):
    value['gradients'] = output[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


# path_s = os.listdir('./picture')
path_s = os.listdir('./sample-imagenet')

def save_cam(cam, image, save_path):
    # save cam
    plt.imshow((cam),cmap='seismic')
    plt.imshow(image, alpha=.5)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # plt.show()
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    plt.close()

    # save segmentation
    plt.imshow(image*threshold(cam)[...,np.newaxis])
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path+'_seg', bbox_inches='tight')
    plt.clf()
    plt.close()

for k, path in enumerate(path_s[:200]):
    try:
        img_path_long = './sample-imagenet/{}'.format(path)
        img = cv2.imread(img_path_long,1)
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_show = cv2.resize(img_show,(224,224))
        img = np.float32(cv2.resize(img, (224,224)))/255

        in_tensor = preprocess_image(img).cuda() if torch.cuda.is_available() else preprocess_image(img)
        XR_CAM, R_CAM, output = model(in_tensor, args.target_layer, [args.target_class])

        if args.target_class == None:
            maxindex = np.argmax(output.data.cpu().numpy())
        else:
            maxindex = args.target_class

        print('{}/{} - {}'.format(k, len(path_s[:200]), index2class[maxindex]))
        output[:, maxindex].sum().backward(retain_graph=True)
        activation = value['activations']  # [1, 2048, 7, 7]
        gradient = value['gradients']  # [1, 2048, 7, 7]
        gradient_2 = gradient ** 2
        gradient_3 = gradient ** 3

        # grad-cam
        gradient_ = torch.mean(gradient, dim=(2, 3), keepdim=True)
        grad_cam = activation * gradient_
        grad_cam = torch.sum(grad_cam, dim=(0, 1))
        grad_cam = torch.clamp(grad_cam, min=0)
        grad_cam = grad_cam.data.cpu().numpy()
        grad_cam = cv2.resize(grad_cam, (224, 224))

        # xgrad-cam
        w = (gradient*activation) / torch.sum(activation, dim=(2,3), keepdim=True).add(1e-8)
        w = torch.sum(w, dim=(2,3), keepdim=True)
        xgrad_cam = activation * w
        xgrad_cam = torch.sum(xgrad_cam, dim=(0,1))
        xgrad_cam = torch.clamp(xgrad_cam, min=0)
        xgrad_cam = xgrad_cam.data.cpu().numpy()
        xgrad_cam = cv2.resize(xgrad_cam, (224, 224))

        # grad-cam++
        alpha_numer = gradient_2
        alpha_denom = 2 * gradient_2 + torch.sum(activation * gradient_3, axis=(2, 3), keepdims=True)  # + 1e-2
        alpha = alpha_numer / alpha_denom
        w = torch.sum(alpha * torch.clamp(gradient, 0), axis=(2, 3), keepdims=True)
        grad_campp = activation * w
        grad_campp = torch.sum(grad_campp, dim=(0, 1))
        grad_campp = torch.clamp(grad_campp, min=0)
        grad_campp = grad_campp.data.cpu().numpy()
        grad_campp = cv2.resize(grad_campp, (224, 224))

        # xrelevance-cam
        XR_CAM = tensor2image(XR_CAM)

        # relevance-cam
        R_CAM = tensor2image(R_CAM)

        # create file directory if not exists
        save_path_parent_dir  = './results-sample-imagenet/{}/{}'.format(img_path_long.split('/')[-1], args.target_layer)
        if not os.path.exists(save_path_parent_dir):
            os.makedirs(save_path_parent_dir)

        # save the cams
        save_path_relevance_cam = './results-sample-imagenet/{}/{}/{}'.format(img_path_long.split('/')[-1], args.target_layer, 'RelevanceCAM')
        save_path_xrelevance_cam = './results-sample-imagenet/{}/{}/{}'.format(img_path_long.split('/')[-1], args.target_layer, 'XRelevanceCAM')
        save_path_xgrad_cam = './results-sample-imagenet/{}/{}/{}'.format(img_path_long.split('/')[-1], args.target_layer, 'XGradCAM')
        save_path_grad_cam = './results-sample-imagenet/{}/{}/{}'.format(img_path_long.split('/')[-1], args.target_layer, 'GradCAM')
        save_path_gradcam_pp = './results-sample-imagenet/{}/{}/{}'.format(img_path_long.split('/')[-1], args.target_layer, 'GradCAM++')

        save_cam(R_CAM, img_show, save_path_relevance_cam)
        save_cam(XR_CAM, img_show, save_path_xrelevance_cam)
        save_cam(xgrad_cam, img_show, save_path_xgrad_cam)
        save_cam(grad_campp, img_show, save_path_gradcam_pp)
        save_cam(grad_cam, img_show, save_path_grad_cam)
    except:
        print('error happens in this iteration')

print('Done')



