import torch
from torch import nn
from torch.nn import functional as F
# transitionで使用されている
class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False, height=480, width=640, dropout=0):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers
        self.height = int(height)
        self.width = int(width)
        self.dropout = dropout

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                            out_channels=self._channels,
                                            stride=1,
                                            height=self.height,
                                            width=self.width,
                                            do_batch_norm=do_batch_norm) for i in range(self._layers)])

    def forward(self,input_res):
        inputs = input_res.clone()
        x = self.res_block(inputs)
        x = F.relu(x)
        # x = F.dropout(x, p=self.dropout)

        input = inputs + x
        return input

# decorderで使用されている
class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, height=480, width=640 , do_batch_norm=False, dropout=0):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self.heihgt = int(height)
        self.width = int(width)
        self._do_batch_norm = do_batch_norm
        self._dropout = dropout

        # conv2d, layer_norm, relu, dropout
        self.conv2d_1 = general_conv2d(in_channels=self._in_channels, 
                                        out_channels=self._out_channels, 
                                        kernel_size=self._kernel_size,
                                        stride=1, 
                                        padding=0,
                                        height=int(height),
                                        width=int(width),
                                        do_batch_norm=self._do_batch_norm,
                                        dropout=self._dropout)

        self.pad = nn.ReflectionPad2d(padding=(int((self._kernel_size-1)/2), int((self._kernel_size-1)/2),
                                        int((self._kernel_size-1)/2), int((self._kernel_size-1)/2)))

        self.predict_flow = general_conv2d(in_channels=self._out_channels, 
                                        out_channels=2, 
                                        kernel_size=1,
                                        stride=1, 
                                        padding=0,
                                        height=int(height),
                                        width=int(width),
                                        dropout=self._dropout,
                                        activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')
        conv = self.pad(conv)
        # conv2d, layer_norm, relu, dropout
        conv = self.conv2d_1(conv)
        # conv2d, layer_norm, tanh, dropout
        flow = self.predict_flow(conv)
        flow = flow * 256

        return torch.cat([conv,flow.clone()], dim=1), flow

# encoder,decorderで使用されている
def general_conv2d(
        in_channels,
        out_channels, 
        kernel_size=3, 
        stride=2, 
        padding=1, 
        height=480,
        width=640,
        image_shape = torch.empty((8, 480, 640), dtype=torch.int64),
        do_batch_norm=False, 
        dropout=0, 
        activation='relu'
        ):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == 'relu':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,
                        stride=stride,padding=padding),
                # nn.LayerNorm(image_shape),
                nn.LayerNorm((out_channels, height , width)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,
                        stride=stride,padding=padding),
                # nn.LayerNorm(out_channels,eps=1e-5,momentum=0.99),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,
                        stride=stride,padding=padding),
                nn.LayerNorm((out_channels, height , width)),
                nn.Tanh(),
                nn.Dropout(p=dropout)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = kernel_size,
                        stride=stride,padding=padding),
                # nn.LayerNorm(out_channels,eps=1e-5,momentum=0.99),
                nn.Tanh(),
                nn.Dropout(p=dropout)
            )
    return conv2d
