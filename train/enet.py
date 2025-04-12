# ===============================================================================
# ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
# URL: https://arxiv.org/abs/1606.02147 by Adam Paszke, Abhishek Chaurasia, et al.
# ===============================================================================

import torch.nn as nn
import torch


class InitialBlock(nn.Module):

    """ Initial block composed of two branches:
        1. the main one that performs regular convoluition with stride 2
        2. the extension one that performas max-pooling

        Both operations done in parallel; concatenating their results allows efficient
        downsampling and expansion

        16 feature maps:
        - main branch outputs 13 of them
        - extesnion branch outputs 3 of them

    """

    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU
        
        #main branch
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=bias)
        
        #extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        
        #inizialization of bach normalization used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        #PReLu layer applied after concatenation of the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        
        #concatenating the two branches
        out = torch.cat((main, ext), 1)
        
        #batch normalization
        out = self.batch_norm(out)
        return self.out_activation(out)

class RegularBottleneck(nn.Module):

    """
        1. Main branch: shortcut connection
        2. Extension branch:
            1. 1x1 convolution (projection) that decreases the number
            of channels by 'internal_ratio'
            2. regular, dilated or asymmetric convolution
            3. 1x1 convolution (expansion) that increases the number
            of channels back to 'channels'
            4. dropout as a regularizer
    """

    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0, dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range.")
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU

        #1x1 project convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        #if convolution is asymmetric, the main convolution splitted in two
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation()
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation()
            )
        
        #1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        
        # PReLU layer applied after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        #main branch shortcut
        main = x
        
        #extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        
        #adding main and ext branches
        out = main + ext
        return self.out_activation(out)

class DownsamplingBottleneck(nn.Module):

    """
        The downsampling bottlenecks downsample the feature map size

        1. Main branch: max pooling with stride 2; indices are saved to
        be used for unpooling later
        2. Extension branch, composed of:
            1. 2x2 convolution with stride 2 (projection) that decreases the number
            of channels by 'internal_ratio'
            2. regular convolution (default 3x3)
            3. 1x1 convolution (expansion) that increases the number
            of channels to 'out_channels'
            4. dropout as a regularizer
    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        self.return_indices = return_indices
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range.")
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        
        #main branch
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        #extension branch
        #2x2 project convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        #regular convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )

        #1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation()
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer applied after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        #main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        
        #extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        if main.is_cuda:
            padding = padding.cuda()
        
        #concatenation
        main = torch.cat((main, padding), 1)
        
        #adding main and ext branches
        out = main + ext
        return self.out_activation(out), max_indices

class UpsamplingBottleneck(nn.Module):

    '''
        The upsampling bottlenecks upsample the feature map

        1. Main branch: 
            1. 1x1 convolution (projection) with stride 1 that
            decrease the number of channels by 'internal_ratio' 
            2. max unpool layr using the max pool indices from the
            corresponding downsampling max pool layer
        2. Extension branch, composed of:
            1. 1x1 convolution with stride 1 (projection) that decreases the number
            of channels by 'internal_ratio'
            2. transposed convolution (default 3x3)
            3. 1x1 convolution (expansion) that increases the number
            of channels to 'out_channels'
            4. dropout as a regularizer
    '''

    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range.")
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        
        #main branch
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        
        #extension branch
        #1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels, internal_channels, kernel_size=2, stride=2, bias=bias
        )
        
        #transposed convolution
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()
        
        #1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        
        # PReLU layer applied after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        #main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)

        #extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        #adding main and ext branches
        out = main + ext
        return self.out_activation(out)

class ENet(nn.Module):
    ''' Generate the ENet model '''
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        #Stage 1 (Encoder)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        
        #Stage 2 (Encoder)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        
        #Stage 3 (Encoder)
        self.regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        
        #Stage 4 (Decoder)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        #Stage 5 (Decoder)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        #initial block
        input_size = x.size()
        x = self.initial_block(x)

        #Stage 1 (Encoder)
        stage1_input_size = x.size()
        x, max_indices1 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        #Stage 2 (Encoder)
        stage2_input_size = x.size()
        x, max_indices2 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        #Stage 3 (Encoder)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        #Stage 4 (Decoder)
        x = self.upsample4_0(x, max_indices2, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        #Stage 5 (Decoder)
        x = self.upsample5_0(x, max_indices1, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=input_size)
        return x
