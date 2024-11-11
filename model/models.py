import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .complexLayers import *

act_fn_by_name = {"tanh": nn.Tanh(), "relu": nn.ReLU(inplace=True), "leakyrelu": nn.LeakyReLU(0.01), "gelu": nn.GELU(), "complex_gelu": ComplexGeLU()}

def get_output_shape(model, input_dim):
    return model(torch.rand(*(input_dim))).data.shape

    
class SSModule(nn.Module):
    def __init__(self, out_channels=19):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(104, out_channels),
        )
        
    def forward(self, input):
        N = len(input)
        out = input.view(N,-1)
        out = self.output(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)

    def forward(self, q, k, v):
        # Assuming shape of x is (B, C, H, W), we need to reshape to (B, H*W, C) for attention
        b, c, h, w = q.size()
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        out, _ = self.attention(q, k, v)
        # Reshape back to (B, C, H, W)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out
    
class Self_ACC(nn.Module):
    def __init__(self, num_classes):
        super(Self_ACC, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.attention1 = AttentionBlock(16) 
        self.bn1 = nn.BatchNorm2d(16, affine=False)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.attention2 = AttentionBlock(16)
        self.bn3 = nn.BatchNorm2d(16, affine=False)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16, affine=False)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.attention3 = AttentionBlock(16)
        self.bn5 = nn.BatchNorm2d(16, affine=False)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(16, affine=False)
        self.maxpool = nn.MaxPool2d((3,1), stride=(3,1))
        self.fc1 = nn.Linear(16*17*1, 512) # Assuming the input size is (52, 1)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial input x [B,1,52,1]
        q = self.conv1(x)
        k = self.conv2(x)
        v = self.conv3(x)
        
        # Attention block 1
        out = self.attention1(q,k,v)
        out = self.bn1(out)
        out = out + x
        out1 = F.relu(out)
        out = self.conv4(out1)
        out = self.bn2(out)
        out = out + out1
        output = F.relu(out)
        
        q = self.conv5(output)
        k = self.conv6(output)
        v = self.conv7(output)
        
        # Attention block 2
        out = self.attention2(q,k,v)
        out = self.bn3(out)
        out = out + output
        out1 = F.relu(out)
        out = self.conv8(out1)
        out = self.bn4(out)
        out = out + out1
        output = F.relu(out)
        
        q = self.conv9(output)
        k = self.conv10(output)
        v = self.conv11(output)
        
        # Attention block 3
        out = self.attention3(q,k,v)
        out = self.bn5(out)
        out = out + output
        out1 = F.relu(out)
        out = self.conv12(out1)
        out = self.bn6(out)
        out = out + out1
        output = F.relu(out)

        # Pooling, flatten, and dense layers
        end = self.maxpool(output)
        end = torch.flatten(end, 1)
        end = F.relu(self.fc1(end))
        end = F.relu(self.fc2(end))
        output = self.fc3(end)
        return output


class ConvNormalization(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides=1, padding='same', activation='selu',
                 kernel_initializer=None, bias_initializer=None, bn=False, name_layer=None):
        super(ConvNormalization, self).__init__()
        
        # Initialize the convolution layer
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters,
                              kernel_size=kernel_size, stride=strides, padding= padding)
        
        # Initialize the batch normalization layer, if requested
        self.bn = bn
        if bn:
            self.bn_layer = nn.BatchNorm2d(filters, affine=False)
        
        # Set the activation function
        if activation == 'selu':
            self.activation = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class att_network(nn.Module):
    def __init__(self, num_classes,input_shape):
        super(att_network, self).__init__()
        self.conv1 = ConvNormalization(1,128, (7,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.conv2 = ConvNormalization(128,128, (7,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.conv3 = ConvNormalization(128,128, (7,1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.conv4 = ConvNormalization(128,128, (5,1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.conv5 = ConvNormalization(128,128, (3,1))
        self.pool5 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))

        self.att_conv = nn.Conv2d(2, 1, kernel_size=(1, 5), padding='same')
        
        self.fc1 = nn.Linear(self._calc_flatten_size(input_shape), 128)
        self.drop1 = nn.AlphaDropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.AlphaDropout(0.2)
        self.fc_logits = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.pool5(self.conv5(x))

        x2_avg = torch.mean(x, dim=1, keepdim=True)
        x2_max, _ = torch.max(x, dim=1, keepdim=True)
        x2_concat = torch.cat([x2_avg, x2_max], dim=1)
        att2 = torch.sigmoid(self.att_conv(x2_concat))
        x_att = x * att2
        x = x + x_att
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = self.drop1(x)
        x = F.selu(self.fc2(x))
        x = self.drop2(x)
        logits = self.fc_logits(x)
        return logits

    def _calc_flatten_size(self, input_shape):
        with torch.no_grad():
            input_tensor = torch.zeros(*input_shape)
            x = self.pool1(self.conv1(input_tensor))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            x = self.pool5(self.conv5(x))
            x2_avg = torch.mean(x, dim=1, keepdim=True)
            x2_max, _ = torch.max(x, dim=1, keepdim=True)
            x2_concat = torch.cat([x2_avg, x2_max], dim=1)
            att2 = torch.sigmoid(self.att_conv(x2_concat))
            x_att = x * att2
            x = x + x_att
            return int(np.prod(x.size()[1:]))

    
class complex_SpatialBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn, dim):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            ComplexConv2d(c_in, c_out["1x1"], kernel_size=1), 
            # ComplexBatchNorm2d(c_out["1x1"]), 
            ComplexLayernorm([c_out["1x1"],dim,1 ],affine=False),
            act_fn
        )

        # 3x3 convolution branch
        self.conv_3x1 = nn.Sequential(
            ComplexConv2d(c_in, c_red["3x1"], kernel_size=1),
            # ComplexBatchNorm2d(c_red["3x1"]),
            ComplexLayernorm([c_red["3x1"],dim,1 ],affine=False),
            # act_fn,
            ComplexConv2d(c_red["3x1"], c_out["3x1"], kernel_size=(3,1), padding=(1,0)),
            # ComplexBatchNorm2d(c_out["3x1"]),
            # CLayerNorm([c_out["3x1"],26,1 ]),
            act_fn,
        )

        # 5x5 convolution branch
        self.conv_5x1 = nn.Sequential(
            ComplexConv2d(c_in, c_red["5x1"], kernel_size=1),
            # ComplexBatchNorm2d(c_red["5x1"]),
            ComplexLayernorm([c_red["5x1"],dim,1 ],affine=False),
            # act_fn,
            ComplexConv2d(c_red["5x1"], c_out["5x1"], kernel_size=(5,1), padding=(2,0)),
            # ComplexBatchNorm2d(c_out["5x1"]),
            # CLayerNorm([c_out["5x1"],26,1 ]),
            act_fn,
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            ComplexMaxPool2d(kernel_size=(3,1), padding=(1,0), stride=1),
            ComplexConv2d(c_in, c_out["max"], kernel_size=1),
            # ComplexBatchNorm2d(c_out["max"]),
            ComplexLayernorm([c_out["max"],dim,1 ],affine=False),
            act_fn,
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x1 = self.conv_3x1(x)
        x_5x1 = self.conv_5x1(x)
        # x_7x1 = self.conv_7x1(x)
        x_max = self.max_pool(x)
        # x_out = torch.cat([x_1x1, x_3x1, x_5x1], dim=1)
        x_out = torch.cat([x_1x1, x_3x1, x_5x1, x_max], dim=1)
        return x_out
   
class complex_DeepCRF(nn.Module):
    def __init__(self, in_channels=1, out_channels=109, d=64, act_fn_name="gelu", **kwargs):
        super().__init__()
        act_fn=act_fn_by_name[act_fn_name]
        partial = 4
        
        self.input_net = nn.Sequential(
            ComplexConv2d(in_channels=in_channels, out_channels=d, kernel_size=(3, 1), stride=1, padding=(1,0)),
            ComplexLayernorm(input_shape=[d,52,1 ],affine=False),
            ComplexGeLU(),

            ComplexConv2d(in_channels=d, out_channels=d, kernel_size=(3,1), stride=1, padding=(1,0)),
            ComplexLayernorm(input_shape=[d,52,1 ],affine=False),
            ComplexGeLU(),
        )
        self.spatial_blocks = nn.Sequential(
            complex_SpatialBlock(
                d,
                c_red={"3x1": int(d/partial), "5x1": int(d/partial)},
                c_out={"1x1": int(d/partial), "3x1": int(d/partial), "5x1": int(d/partial), "max": int(d/partial)},
                act_fn=ComplexGeLU(),
                dim = 52
            ),
            complex_SpatialBlock(
                d,
                c_red={"3x1": int(d/partial), "5x1": int(d/partial)},
                c_out={"1x1": int(d/partial), "3x1": int(d/partial), "5x1": int(d/partial), "max": int(d/partial)},
                act_fn=ComplexGeLU(),
                dim = 52
            ),
    
            ComplexMaxPool2d(kernel_size=(3,1), stride=2, ceil_mode=True),
        )
        self.output_net = nn.Sequential(
            ComplexLinear(d*52//2, 52),
        )
        self.classifer = nn.Sequential(
            ComplexGeLU(),
            ComplexDropout(p=0.5),
            ComplexLinear(52, out_channels),
        )

    def forward(self, x):
        N = len(x)
        x = x[:,:,:,0] + 1j*x[:,:,:,1]
        x = x[:,:,:,None]
        x = self.input_net(x)
        x = self.spatial_blocks(x).view(N, -1)
        x = self.output_net(x)
        x = self.classifer(x)
        x = x.abs()
        return x 

class DeepCRFConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128, in_channels=1, d=64, act_fn_name="gelu"):
        super().__init__()
        self.expected_input_shape = (1, 1, 52, 2)
        self.encoder = DeepCRF_Encoder(in_channels=in_channels, out_channels=d*52//4, d=d, act_fn_name=act_fn_name)
        dim = np.prod(list(get_output_shape(self.encoder, self.expected_input_shape)))
        if head == 'linear':
            self.head = nn.Sequential(
                nn.Linear(dim, feat_dim),
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class DeepCRF(nn.Module):
    """encoder + classifier"""
    def __init__(self, in_channels=1, out_channels=19, d=64, act_fn_name="gelu"):
        super().__init__()
        self.expected_input_shape = (1, 1, 52, 2)
        act_fn=act_fn_by_name[act_fn_name]
        self.encoder = DeepCRF_Encoder(in_channels=in_channels, out_channels=d*52//4, d=d, act_fn_name=act_fn_name)
        dim = np.prod(list(get_output_shape(self.encoder, self.expected_input_shape)))
        self.classifier = nn.Sequential(
            act_fn,
            nn.Linear(dim, out_channels),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x)) 

    
class SpatialBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn, H, W):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), 
            nn.LayerNorm([c_out["1x1"],H,W],elementwise_affine=False), 
            act_fn
        )

        # 3x3 convolution branch
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x1"], kernel_size=1),
            act_fn,
            nn.Conv2d(c_red["3x1"], c_out["3x1"], kernel_size=(3,1), padding=(1,0)),
            nn.LayerNorm([c_out["3x1"],H,W],elementwise_affine=False),
            act_fn,
        )

        # 5x5 convolution branch
        self.conv_5x1 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x1"], kernel_size=1),
            act_fn,
            nn.Conv2d(c_red["5x1"], c_out["5x1"], kernel_size=(5,1), padding=(2,0)),
            nn.LayerNorm([c_out["5x1"],H,W],elementwise_affine=False),
            act_fn,
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,1), padding=(1,0), stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.LayerNorm([c_out["max"],H,W],elementwise_affine=False),
            act_fn,
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x1 = self.conv_3x1(x)
        x_5x1 = self.conv_5x1(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x1, x_5x1, x_max], dim=1)
        return x_out


class DeepCRF_Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=109, d=64, act_fn_name="leakyrelu", **kwargs):
        super().__init__()
        act_fn=act_fn_by_name[act_fn_name]
        partial = 4
        self.input_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=d, kernel_size=(3, 1), stride=1, padding=(1,0)),
            nn.LayerNorm([d,52,2 ], elementwise_affine=False),
            act_fn,
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=(3,2), stride=1, padding=(1,0)),
            nn.LayerNorm([d,52,1],elementwise_affine=False),
            act_fn,
        )

        self.spatial_blocks = nn.Sequential(
            SpatialBlock(
                d,
                c_red={"3x1": int(d/partial), "5x1": int(d/partial)},
                c_out={"1x1": int(d/partial), "3x1": int(d/partial), "5x1": int(d/partial), "max": int(d/partial)},
                act_fn=act_fn,
                H = 52,
                W = 1,
            ),
            SpatialBlock(
                d,
                c_red={"3x1": int(d/partial), "5x1": int(d/partial)},
                c_out={"1x1": int(d/partial), "3x1": int(d/partial), "5x1": int(d/partial), "max": int(d/partial)},
                act_fn=act_fn,
                H = 52,
                W = 1,
            ),
            nn.MaxPool2d((3,1), stride=2, ceil_mode=True), # N*D*3*1
        )
        self.output_net = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d*26, 52)
        )

    def forward(self, x):
        N = len(x)
        x = self.input_net(x)
        x = self.spatial_blocks(x).view(N, -1)
        x = self.output_net(x)
        return x 
