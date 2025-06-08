import math
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm import create_model
from timm import create_model
from timm.models.layers import DropPath, trunc_normal_
from diffusion.gaussian_diffusion import *
from torchvision.models import resnet18,resnet34,resnet50
from vig import *
import math
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
class DOConv2d(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride,
                 padding, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.d_diag = Parameter(d_diag, requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)

        else:

            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
_pair = _ntuple(2)

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
class DWConv(nn.Module):
    def __init__(self, dim=256):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class DyGraphConv2d(GraphConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()

def get_2d_relative_pos_embed(embed_dim, grid_size):

    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos

class Grapher(nn.Module):

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=256, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x

class GCN(torch.nn.Module):
    def __init__(self, opt):
        super(GCN, self).__init__()
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        self.img_size = opt.img_size
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        max_dilation = 49 // max(num_knn)
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], self.img_size // 4, self.img_size // 4))
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))

            for j in range(blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        DOConv2d(channels[i], channels[i] // 2, kernel_size=1,padding=1),
                        nn.BatchNorm2d(channels[i] // 2),
                        nn.ReLU(),
                        DOConv2d(channels[i] // 2, channels[i] // 2, kernel_size=3,padding=1),
                        nn.BatchNorm2d(channels[i] // 2),
                        nn.ReLU(),
                        nn.Conv2d(channels[i] // 2, channels[i] // 2, kernel_size=1, stride=1, padding=0, bias=bias),
                        nn.BatchNorm2d(channels[i] // 2),
                        nn.ReLU(),
                    ),
                    Grapher(channels[i] // 2, num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                            bias, stochastic, epsilon, n=256, drop_path=dpr[idx],
                            relative_pos=True),
                    FFN(channels[i] // 2, channels[i] * 4, act=act, drop_path=dpr[idx])
                ]
                idx += 1
        self.backbone = nn.Sequential(*self.backbone)
        self.additional_conv = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i] , kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(channels[i] ),
            nn.ReLU(),
            DOConv2d(channels[i] , channels[i] , kernel_size=1, padding=1),
            nn.BatchNorm2d(channels[i] ),
            nn.ReLU(),
            DOConv2d(channels[i] , channels[i] , kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[i]),
            nn.ReLU(),
            nn.Conv2d(channels[i], channels[i], kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(channels[i]),
            nn.ReLU(),
        ])

        self.model_init()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        out = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            out.append(x)
        for conv_layer in self.additional_conv:
            x = conv_layer(x)
        out.append(x)
        return out

def vig(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=2, drop_path_rate=0.0, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.channels = [64, 128, 256, 512]
            self.n_classes = num_classes
            self.img_size = 256
    opt = OptInit(**kwargs)
    model = GCN(opt)
    model.default_cfg = default_cfgs['vig']
    return model

class GDSN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_images = 0
        self.model = Resnet()
        self.pvig = vig(pretrained=True)
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         )
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                )
        self.sampler = UniformSampler(1000)
        self.num_images = 0
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def backbone(self, image=None):
        outputs = []
        features = self.pvig(image)
        x0 = features[0]
        x1 = features[1]
        x2 = features[2]
        x3 = features[3]
        # outputs.append(x0)
        # outputs.append(x1)
        # outputs.append(x2)
        # outputs.append(x3)
        # self.vis_feature(outputs)
        return [x0, x1, x2, x3]
    def forward(self, imageA=None, imageB=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        elif pred_type == "dd_sample":
            embeddingsA = self.backbone(imageA)
            embeddingsB = self.backbone(imageB)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                model_kwargs={"imageA": imageA, "imageB": imageB, "embeddingsA": embeddingsA, "embeddingsB": embeddingsB})
            sample_out = sample_out["pred_xstart"]
            return sample_out
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class EA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class SNCM(nn.Module):
    def __init__(self, dim_x_noise, dim_x_feature):
        super().__init__()

        self.pre_project = nn.Conv2d(dim_x_noise, dim_x_feature, 1)

        group_size = dim_x_noise // 2

        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            Conv(op_channel=group_size),
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            Conv(op_channel=group_size),
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            Conv(op_channel=group_size),
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            Conv(op_channel=group_size),
        )

        self.final_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_x_noise * 2, data_format='channels_first'),
            CoordAtt(inp=dim_x_noise * 2, oup=dim_x_noise)

        )
        self.ema = EA(dim_x_noise)

    def forward(self, x_noise, x_feature):
        x_f_temp = x_feature
        x_noise = self.pre_project(x_noise)

        x_noise = torch.chunk(x_noise, 4, dim=1)
        x_feature = torch.chunk(x_feature, 4, dim=1)

        x0 = self.g0(torch.cat((x_noise[0], x_feature[0]), dim=1))
        x1 = self.g1(torch.cat((x_noise[1], x_feature[1]), dim=1))
        x2 = self.g2(torch.cat((x_noise[2], x_feature[2]), dim=1))
        x3 = self.g3(torch.cat((x_noise[3], x_feature[3]), dim=1))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.final_conv(x)
        out = self.ema(out)

        return out
import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()

        self.inp = inp
        self.oup = oup

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, oup // reduction)
        self.conv1 = nn.Conv2d(oup, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        if self.inp != self.oup:
            self.change_channel = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.inp != self.oup:
            x = self.change_channel(x)
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return x + out

class UPM(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(UPM, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True)
        )
        self.coordatt = CoordAtt(self.inchannel, self.inchannel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.inchannel + self.outchannel, self.outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)
        x_copy = x1.clone()
        x1 = self.conv1(x1)
        x1 = self.coordatt(x1)

        x_f = x1 + x_copy
        return x_f

class HDFM(nn.Module):
    def __init__(self):
        super(HDFM, self).__init__()

        self.upm1 = UPM(512, 256)
        self.upm2 = UPM(256, 128)
        self.upm3 = UPM(128, 64)

        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x1, x2, x3, x4):
        # refine
        r1 = self.refine1(x1, x2)
        r2 = self.refine2(r1, x3)
        x = self.refine3(r2, x4)
        p2 = x
        p3 = self.conv_pool1(x)
        p4 = self.conv_pool2(x)
        p5 = self.conv_pool3(x)

        return p5, p4, p3, p2
from models.DDSM import *
class Resnet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=4, num_classes=6, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.2,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6,i=0),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super(Resnet, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        cur += depths[0]
        cur += depths[1]
        cur += depths[2]
        self.resnet = CustomResNet()
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128, 512),
            torch.nn.Linear(512, 512),
        ])
        self.temb_3 = torch.nn.Linear(512, 64)
        self.temb_2 = torch.nn.Linear(512, 128)
        self.temb_1 = torch.nn.Linear(512, 256)
        self.temb_4 = torch.nn.Linear(512, 512)
        self.Classifier = Decoder(embedding_dim=64)
        self.sncm1 = SNCM(64,64)
        self.sncm2 = SNCM(128,128)
        self.sncm3 = SNCM(256, 256)
        self.sncm4 = SNCM(512, 512)

        self.DDSM_1_1 = DDSM_1(64,64,64)
        self.DDSM_1_2 = DDSM_1(128,128,128)
        self.DDSM_1_3 = DDSM_1(256,256,256)
        self.DDSM_1_4 = DDSM_1(512,512,512)

        self.DDSM_2_1 = DDSM_2(512, 256, 256)
        self.DDSM_2_2 = DDSM_2(256, 128, 128)
        self.DDSM_2_3 = DDSM_2(128, 64, 64)

        self.down1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.HDFM1 = HDFM()
        self.HDFM2 = HDFM()
        self.num_images = 0

    def denoising_stage(self, x: torch.Tensor, embeddings=None, image=None, temb=None,i=0):
        if image is not None:
            x0 = torch.cat([image, x], dim=1)
        outputs = []
        x0, x1, x2, x3 = self.resnet(x0)
        H, W = x0.shape[2:]
        # print(x0.shape)
        x0 = x0 + self.temb_3(nonlinearity(temb))[:, :, None, None]
        # outputs.append(x0)
        x0 = self.sncm1(x0, embeddings[0])
        x1 = x1 + self.temb_2(nonlinearity(temb))[:, :, None, None]
        # outputs.append(x1)
        x1 = self.sncm2(x1, embeddings[1])
        x2 = x2 + self.temb_1(nonlinearity(temb))[:, :, None, None]
        # outputs.append(x2)
        x2 = self.sncm3(x2, embeddings[2])
        x3 = x3 + self.temb_4(nonlinearity(temb))[:, :, None, None]
        # outputs.append(x3)
        x3 = self.sncm4(x3, embeddings[3])
        return [x0, x1, x2, x3]

    def forward(self, x: torch.Tensor, t, embeddingsA=None, embeddingsB=None, imageA=None, imageB=None,):
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)

        res1 = self.denoising_stage(x, embeddingsA, imageA, temb)
        res2 = self.denoising_stage(x, embeddingsB, imageB, temb)

        Diff1 = torch.abs(res1[0] - res2[0])
        Diff2 = torch.abs(res1[1] - res2[1])
        Diff3 = torch.abs(res1[2] - res2[2])
        Diff4 = torch.abs(res1[3] - res2[3])

        Diff4, Diff3, Diff2, Diff1 = self.HDFM1(Diff4, Diff3, Diff2, Diff1)

        Add1 = torch.cat([res1[0], res2[0]], dim=1)
        Add2 = torch.cat([res1[1], res2[1]], dim=1)
        Add3 = torch.cat([res1[2], res2[2]], dim=1)
        Add4 = torch.cat([res1[3], res2[3]], dim=1)

        Add1 = self.down1(Add1)
        Add2 = self.down2(Add2)
        Add3 = self.down3(Add3)
        Add4 = self.down4(Add4)

        Add4, Add3, Add2, Add1 = self.HDFM2(Add4, Add3, Add2, Add1)
        Fusion1 = self.DDSM_1_1(Add1,Diff1)

        Fusion2 = self.DDSM_1_2(Add2,Diff2)

        Fusion3 = self.DDSM_1_3(Add3,Diff3)

        Fusion4 = self.DDSM_1_4(Add4,Diff4)

        x33 = self.DDSM_2_1(Fusion4, Fusion3)
        x33 = x33 + self.temb_1(nonlinearity(temb))[:, :, None, None]
        x22 = self.DDSM_2_2(x33, Fusion2)
        x22 = x22 + self.temb_2(nonlinearity(temb))[:, :, None, None]
        x11 = self.DDSM_2_3(x22, Fusion1)
        x11 = x11 + self.temb_3(nonlinearity(temb))[:, :, None, None]
        pred = self.Classifier(x11)
        return pred
