import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvBlock
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y
# Encoding block
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.down = nn.MaxPool3d(2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        return y, y_conv
# Decoding block
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        return y, y_conv

def concatenate(x1, x2):
    diffZ = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    diffX = x2.size()[4] - x1.size()[4]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2,
                    diffZ // 2, diffZ - diffZ//2))        
    y = torch.cat([x2, x1], dim=1)
    return y

class sub_encoder(nn.Module):
    def __init__(self, in_ch, base_ch):
        super(sub_encoder, self).__init__()
        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch*2)
        self.enc3 = enc_block(base_ch*2, base_ch*4)
        self.enc4 = enc_block(base_ch*4, base_ch*8)

    def forward(self, x):
        y, enc_conv_1 = self.enc1(x)
        y, enc_conv_2 = self.enc2(y)
        y, enc_conv_3 = self.enc3(y)
        y, enc_conv_4 = self.enc4(y)
        return y, enc_conv_1, enc_conv_2, enc_conv_3, enc_conv_4


class sub_decoder(nn.Module):
    def __init__(self, base_ch, cls_num):
        super(sub_decoder, self).__init__()
        self.dec1 = dec_block(base_ch*8,  base_ch*8)
        self.dec2 = dec_block(base_ch*16, base_ch*4)
        self.dec3 = dec_block(base_ch*8,  base_ch*2)
        self.dec4 = dec_block(base_ch*4,  base_ch)
        self.lastconv = double_conv(base_ch*2, base_ch)
        self.outconv = nn.Conv3d(base_ch, cls_num+1, 1)
        self.softmax = nn.Softmax(dim=1)

        self.clssifier1 = nn.Sequential(
            nn.Conv3d(base_ch*8, cls_num+1, 1),
            nn.Upsample(scale_factor=16, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier2 = nn.Sequential(
            nn.Conv3d(base_ch*4, cls_num+1, 1),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier3 = nn.Sequential(
            nn.Conv3d(base_ch*2, cls_num+1, 1),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier4 = nn.Sequential(
            nn.Conv3d(base_ch*1, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x, e1, e2, e3, e4):
        y, ds1 = self.dec1(x)
        y1 = self.clssifier1(ds1)
        y, ds2 = self.dec2(concatenate(y, e4))
        y2 = self.clssifier2(ds2)
        y, ds3 = self.dec3(concatenate(y, e3))
        y3 = self.clssifier3(ds3)
        y, ds4 = self.dec4(concatenate(y, e2))
        y4 = self.clssifier4(ds4)
        y = self.lastconv(concatenate(y, e1))
        y = self.outconv(y)
        output = self.softmax(y)

        return output, y4, y3, y2, y1

class aux_decoder(nn.Module):
    def __init__(self, base_ch, cls_num):
        super(aux_decoder, self).__init__()
        self.clssifier4 = nn.Sequential(
            double_conv(base_ch*8, base_ch),
            nn.Conv3d(base_ch, cls_num+1, 1),
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier3 = nn.Sequential(
            double_conv(base_ch*4, base_ch),
            nn.Conv3d(base_ch, cls_num+1, 1),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier2 = nn.Sequential(
            double_conv(base_ch*2, base_ch),
            nn.Conv3d(base_ch, cls_num+1, 1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.clssifier1 = nn.Sequential(
            double_conv(base_ch, base_ch),
            nn.Conv3d(base_ch, cls_num+1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, e1, e2, e3, e4):
        y4 = self.clssifier4(e4)
        y3 = self.clssifier3(e3)
        y2 = self.clssifier2(e2)
        y1 = self.clssifier1(e1)

        return y1, y2, y3, y4

class MENUNet(nn.Module):
    def __init__(self, in_ch, base_ch, cls_num):
        super(MENUNet, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cls_num = cls_num

        self.sub_encoders = nn.ModuleList([sub_encoder(in_ch, base_ch) for i in range(cls_num)])
        self.aux_decoder = aux_decoder(base_ch, 1)

        self.global_decoder = sub_decoder(cls_num*base_ch, cls_num)

    def forward(self, x, node_enabled_encoders=None):
        if node_enabled_encoders is None:
            y = []
            enc_conv_1 = []
            enc_conv_2 = []
            enc_conv_3 = []
            enc_conv_4 = []
            for i in range(self.cls_num):
                e, e1, e2, e3, e4 = self.sub_encoders[i](x)
                y.append(e)
                enc_conv_1.append(e1)
                enc_conv_2.append(e2)
                enc_conv_3.append(e3)
                enc_conv_4.append(e4)

            y = torch.cat(y, dim=1)
            enc_conv_1 = torch.cat(enc_conv_1, dim=1)
            enc_conv_2 = torch.cat(enc_conv_2, dim=1)
            enc_conv_3 = torch.cat(enc_conv_3, dim=1)
            enc_conv_4 = torch.cat(enc_conv_4, dim=1)

            if self.training:
                output, y1, y2, y3, y4 = self.global_decoder(y, enc_conv_1, enc_conv_2, enc_conv_3, enc_conv_4)
                return [output, y1, y2, y3, y4]
            else:
                output, _, _, _, _ = self.global_decoder(y, enc_conv_1, enc_conv_2, enc_conv_3, enc_conv_4)
                return output
        else:
            output = []
            for encoder_id in node_enabled_encoders:
                _, e1, e2, e3, e4 = self.sub_encoders[encoder_id](x)
                y1, y2, y3, y4 = self.aux_decoder(e1, e2, e3, e4)
                output.append([y1, y2, y3, y4])
            return output

    def get_s1_parameters(self, node_enabled_encoders):
        params = list(self.global_decoder.parameters())
        for encoder_id in node_enabled_encoders:
            params += list(self.sub_encoders[encoder_id].parameters())
        return params

    def get_s2_parameters(self, node_enabled_encoders):
        params = list(self.aux_decoder.parameters())
        for encoder_id in node_enabled_encoders:
            params += list(self.sub_encoders[encoder_id].parameters())
        return params

    def description(self):
        return 'Multi-encoder U-Net (input channel = {0:d}) for {1:d}-organ segmentation (base channel = {2:d})'.format(self.in_ch, self.cls_num, self.base_ch)