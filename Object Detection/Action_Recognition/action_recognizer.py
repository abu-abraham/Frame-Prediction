import torch
from torch.autograd import Variable


import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import random


import numbers
import collections
import numpy as np
from PIL import Image, ImageOps


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400, last_fc=True):
        self.last_fc = last_fc

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

try:
    import accimage
except ImportError:
    accimage = None


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float()

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    print("video loader")
    print(video_loader)
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(video_path, sample_duration):
    dataset = []

    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration
    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        dataset.append(sample_i)
    return dataset


class Video(data.Dataset):
    def __init__(self, images,
                 spatial_transform=None, temporal_transform=None,
                 sample_duration=16, get_loader=get_default_video_loader):

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        #self.loader = get_loader()
        self.images = images

    def __getitem__(self, index):
        clip = self.images
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip

    def __len__(self):
        return 1


class ActionRecognizer:
    sample_size = 112
    class_names = []

    
    def __init__(self):
        self.model = resnet34(num_classes=400, shortcut_type='A', sample_size=112, sample_duration=5, last_fc=True)
        self.class_names = ['abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream', 'archery', 'arm wrestling', 'arranging flowers', 'assembling computer', 'auctioning', 'baby waking up', 'baking cookies', 'balloon blowing', 'bandaging', 
            'barbequing', 'bartending', 'beatboxing', 'bee keeping', 'belly dancing', 'bench pressing', 'bending back', 'bending metal', 'biking through snow', 
            'blasting sand', 'blowing glass', 'blowing leaves', 'blowing nose', 'blowing out candles', 'bobsledding', 'bookbinding', 'bouncing on trampoline', 
            'bowling', 'braiding hair', 'breading or breadcrumbing', 'breakdancing', 'brush painting', 'brushing hair', 'brushing teeth', 'building cabinet', 'building shed', 
            'bungee jumping', 'busking', 'canoeing or kayaking', 'capoeira', 'carrying baby', 'cartwheeling', 'carving pumpkin', 'catching fish', 'catching or throwing baseball', 
            'catching or throwing frisbee', 'catching or throwing softball', 'celebrating', 'changing oil', 'changing wheel', 'checking tires', 'cheerleading', 'chopping wood', 'clapping', 
            'clay pottery making', 'clean and jerk', 'cleaning floor', 'cleaning gutters', 'cleaning pool', 'cleaning shoes', 'cleaning toilet', 'cleaning windows', 'climbing a rope', 'climbing ladder', 'climbing tree', 'contact juggling', 'cooking chicken', 
            'cooking egg', 'cooking on campfire', 'cooking sausages', 'counting money', 'country line dancing', 'cracking neck', 'crawling baby', 'crossing river', 'crying', 'curling hair', 'cutting nails', 'cutting pineapple', 'cutting watermelon', 'dancing ballet', 
            'dancing charleston', 'dancing gangnam style', 'dancing macarena', 'deadlifting', 'decorating the christmas tree', 'digging', 'dining', 'diving cliff', 'dodgeball', 'doing aerobics', 'doing laundry', 'doing nails', 'drawing', 'dribbling basketball', 'drinking', 'drinking beer', 
            'drinking shots', 'driving car', 'driving tractor', 'drop kicking', 'drumming fingers', 'dunking basketball', 'dying hair', 'eating burger', 'eating cake', 'eating carrots', 'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream', 'eating spaghetti', 'eating watermelon', 
            'egg hunting', 'exercising arm', 'exercising with an exercise ball', 'extinguishing fire', 'faceplanting', 'feeding birds', 'feeding fish', 'feeding goats', 'filling eyebrows', 'finger snapping', 'fixing hair', 'flipping pancake', 'flying kite', 'folding clothes', 'folding napkins', 'folding paper', 
            'front raises', 'frying vegetables', 'garbage collecting', 'gargling', 'getting a haircut', 'getting a tattoo', 'giving or receiving award', 'golf chipping', 'golf driving', 'golf putting', 'grinding meat', 'grooming dog', 'grooming horse', 'gymnastics tumbling', 'hammer throw', 'headbanging', 'headbutting', 
            'high jump', 'high kick', 'hitting baseball', 'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging', 'hula hooping', 'hurdling', 'hurling (sport)', 'ice climbing', 'ice fishing', 'ice skating', 'ironing', 'javelin throw', 'jetskiing', 'jogging', 'juggling balls', 'juggling fire', 
            'juggling soccer ball', 'jumping into pool', 'jumpstyle dancing', 'kicking field goal', 'kicking soccer ball', 'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing', 'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich', 'making bed', 'making jewelry', 'making pizza', 
            'making snowman', 'making sushi', 'making tea', 'marching', 'massaging back', 'massaging feet', 'massaging legs', "massaging person's head", 'milking cow', 'mopping floor', 'motorcycling', 'moving furniture', 'mowing lawn', 'news anchoring', 'opening bottle', 'opening present', 'paragliding', 'parasailing',
            'parkour', 'passing American football (in game)', 'passing American football (not in game)', 'peeling apples', 'peeling potatoes', 'petting animal (not cat)', 'petting cat', 'picking fruit', 'planting trees', 'plastering', 'playing accordion', 'playing badminton', 'playing bagpipes', 'playing basketball', 
            'playing bass guitar', 'playing cards', 'playing cello', 'playing chess', 'playing clarinet', 'playing controller', 'playing cricket', 'playing cymbals', 'playing didgeridoo', 'playing drums', 'playing flute', 'playing guitar', 'playing harmonica', 'playing harp', 'playing ice hockey', 'playing keyboard', 
            'playing kickball', 'playing monopoly', 'playing organ', 'playing paintball', 'playing piano', 'playing poker', 'playing recorder', 'playing saxophone', 'playing squash or racquetball', 'playing tennis', 'playing trombone', 'playing trumpet', 'playing ukulele', 'playing violin', 'playing volleyball', 'playing xylophone', 
            'pole vault', 'presenting weather forecast', 'pull ups', 'pumping fist', 'pumping gas', 'punching bag', 'punching person (boxing)', 'push up', 'pushing car', 'pushing cart', 'pushing wheelchair', 'reading book', 'reading newspaper', 'recording music', 'riding a bike', 'riding camel', 'riding elephant', 'riding mechanical bull', 
            'riding mountain bike', 'riding mule', 'riding or walking with horse', 'riding scooter', 'riding unicycle', 'ripping paper', 'robot dancing', 'rock climbing', 'rock scissors paper', 'roller skating', 'running on treadmill', 'sailing', 'salsa dancing', 'sanding floor', 'scrambling eggs', 'scuba diving', 'setting table', 'shaking hands', 
            'shaking head', 'sharpening knives', 'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep', 'shining shoes', 'shooting basketball', 'shooting goal (soccer)', 'shot put', 'shoveling snow', 'shredding paper', 'shuffling cards', 'side kick', 'sign language interpreting', 'singing', 'situp', 'skateboarding', 'ski jumping', 
            'skiing (not slalom or crosscountry)', 'skiing crosscountry', 'skiing slalom', 'skipping rope', 'skydiving', 'slacklining', 'slapping', 'sled dog racing', 'smoking', 'smoking hookah', 'snatch weight lifting', 'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling', 'somersaulting', 'spinning poi', 'spray painting', 
            'spraying', 'springboard diving', 'squat', 'sticking tongue out', 'stomping grapes', 'stretching arm', 'stretching leg', 'strumming guitar', 'surfing crowd', 'surfing water', 'sweeping floor', 'swimming backstroke', 'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing', 'swinging legs', 'swinging on something', 'sword fighting', 
            'tai chi', 'taking a shower', 'tango dancing', 'tap dancing', 'tapping guitar', 'tapping pen', 'tasting beer', 'tasting food', 'testifying', 'texting', 'throwing axe', 'throwing ball', 'throwing discus', 'tickling', 'tobogganing', 'tossing coin', 'tossing salad', 'training dog', 'trapezing', 'trimming or shaving beard', 'trimming trees', 'triple jump', 
            'tying bow tie', 'tying knot (not on a tie)', 'tying tie', 'unboxing', 'unloading truck', 'using computer', 'using remote controller (not gaming)', 'using segway', 'vault', 'waiting in line', 'walking the dog', 'washing dishes', 'washing feet', 'washing hair', 'washing hands', 'water skiing', 'water sliding', 'watering plants', 'waxing back', 'waxing chest',
            'waxing eyebrows', 'waxing legs', 'weaving basket', 'welding', 'whistling', 'windsurfing', 'wrapping present', 'wrestling', 'writing', 'yawning', 'yoga', 'zumba']


        #Load model here

    def identifyAction(self, image_array):
        print(image_array)
        spatial_transform = Compose([Scale(self.sample_size),
                                 CenterCrop(self.sample_size),
                                 ToTensor(),
                                 Normalize([114.7748, 107.7354, 99.475], [1, 1, 1])])
        temporal_transform = LoopPadding(5)
        data = Video(image_array, spatial_transform=spatial_transform, temporal_transform=temporal_transform, sample_duration=5)
        data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        video_outputs = []
        video_segments = []
        for i, (inputs) in enumerate(data_loader):
            inputs = Variable(inputs, volatile=True)
            print(inputs.shape)
            outputs = self.model(inputs)
            video_outputs.append(outputs.cpu().data)
            break

        video_outputs = torch.cat(video_outputs)
        _, max_indices = video_outputs.max(dim=1)
        print(self.class_names[max_indices[0]])
        return "Runnin"
	
