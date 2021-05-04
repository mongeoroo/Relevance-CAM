import numpy as np
import torch
from torch.autograd import Variable
import cv2
from imagenet_index import index2class
import torch.nn.functional as F

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def min_max(x):

    x -= x.min()
    x /= x.max()
    return x


def z_norm(x):

    x = (x-x.mean())/x.std()

    return x

def threshold(x):
    mean_ = x.mean()
    std_ = x.std()
    thresh = mean_ +std_
    x = (x>thresh)

    return x


def hm_to_rgb(R, cmap = 'bwr', normalize = True):
    import matplotlib.cm
    import cv2
    cmap = eval('matplotlib.cm.{}'.format(cmap))
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    R = R
    R = cv2.resize(R, (224,224))
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    return rgb

class GradCAM(object):

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].sum()
        else:
            score = logit[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        a = self.activations['value'].detach().cpu().numpy()
        g = self.gradients['value'].detach().cpu().numpy()
        a = np.transpose(a[0], (1, 2, 0)).copy()
        g = np.transpose(g[0], (1, 2, 0)).copy()
        g_ = np.mean(g, axis=(0, 1), keepdims=True).copy()


        grad_cam =a*g_
        grad_cam = np.sum(np.maximum(grad_cam,0),axis=-1)

        return grad_cam


    def __call__(self, input, class_idx=None):
        return self.forward(input, class_idx)


class GradCAM_multi(object):

    def __init__(self, model, target_layer, gradients_flag = False, g_flag = True, g_pp_flag = True):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.g_flag = g_flag
        self.g_pp_flag = g_pp_flag
        self.gradients_flag = gradients_flag
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].sum()
        else:
            score = logit[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        a = self.activations['value'].detach().cpu().numpy()
        if self.g_pp_flag == False and self.g_flag == False and self.gradients_flag == False:
            return a[0]
        g = self.gradients['value'].detach().cpu().numpy()
        a = np.transpose(a[0], (1, 2, 0)).copy()
        g = np.transpose(g[0], (1, 2, 0)).copy()
        g_ = np.mean(g, axis=(0, 1), keepdims=True).copy()

        grad_cam, grad_cam_pp = 0,0

        if self.g_pp_flag:
            g_2 = g**2
            g_3 = g**3
            alpha_numer = g_2
            alpha_denom = 2*g_2 + np.sum(a*g_3,axis=(0,1),keepdims=True) #+ 1e-2

            alpha = alpha_numer/ alpha_denom

            w = np.sum(alpha*np.maximum(g,0),axis=(0,1),keepdims=True)

            grad_cam_pp = np.maximum(w * a,0)
            grad_cam_pp = np.sum(grad_cam_pp, axis=-1)
            grad_cam_pp = cv2.resize(grad_cam_pp, (224, 224))

        if self.g_flag:

            grad_cam =a*g_
            grad_cam = np.sum(np.maximum(grad_cam,0),axis=-1)
            grad_cam = cv2.resize(grad_cam,(224,224))

        if self.gradients_flag: return g_[0,0], a, grad_cam, grad_cam_pp
        else:                   return a, grad_cam, grad_cam_pp

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

class Activation(object):
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = dict()
        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        with torch.no_grad():
            logit = self.model(input)

        a = self.activations['value'].detach().cpu().numpy()
        a = a[0]
        a = np.transpose(a,(1,2,0))
        return a

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class ScoreCAM(object):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        def backward_hook(module,input,output):
            self.gradients['value'] = output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        score_weight = []
        # predication on raw input
        logit = self.model(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

              if saliency_map.max() == saliency_map.min():
                  score_weight.append(0)
                  continue

              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # how much increase if keeping the highlighted region
              # predication on masked input
              output = self.model(input * norm_saliency_map)
              output = F.softmax(output)
              score = output[0][predicted_class]

              score_saliency_map += score * saliency_map
              score_weight.append(score.detach().cpu().numpy())

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map, score_weight

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)



def enlarge_image(img, scaling = 3):
    if scaling < 1 or not isinstance(scaling,int):
        print ('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape
        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]
    elif len(img.shape) == 3:
        H,W,D = img.shape
        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]
    return out


def hm_to_rgb(R, scaling = 1, cmap = 'seismic', normalize = True):
    import cv2
    import matplotlib
    '''R shape: (224,224)'''
    cmap = eval('matplotlib.cm.{}'.format(cmap))
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    R = R
    R = cv2.resize(R,(224,224))
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    return rgb


def LRP(output, max_index):
    if max_index == None:
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print('Pred cls : '+str(pred))
        max_index = pred.squeeze().cpu().numpy()

    tmp = np.zeros(output.shape)
    tmp[:,max_index] = 1
    T = tmp
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt

def SGLRP(output, max_index = None):
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output)
    if max_index == None:
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print('Pred cls : '+str(pred))
        max_index = pred.squeeze().cpu().numpy()

    output = output.detach().cpu().numpy()
    tmp = output.copy()
    y_t = tmp[:,max_index].copy()
    tmp *= y_t
    tmp[:, max_index] = 0

    Tn = tmp
    Tn = torch.from_numpy(Tn).type(torch.FloatTensor)
    Tn = Variable(Tn).cuda()

    tmp = np.zeros(output.shape)
    tmp[:,max_index] = y_t*(1-y_t)
    T = tmp
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()

    return Tt, Tn

def CLRP(output, max_index = None):
    if max_index == None:
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print('Pred cls : '+str(pred))
        max_index = pred.squeeze().cpu().numpy()

    tmp = np.ones(output.shape)
    tmp *= 1/1000
    tmp[:,max_index] = 0
    Tn = tmp
    with torch.no_grad():
        Tn = torch.from_numpy(Tn).type(torch.FloatTensor)
        Tn = Variable(Tn).cuda()

    tmp = np.zeros(output.shape)
    tmp[:,max_index] = 1
    T = tmp
    with torch.no_grad():
        T = torch.from_numpy(T).type(torch.FloatTensor)
        Tt = Variable(T).cuda()


    return Tt,Tn

def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt

