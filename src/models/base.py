import torch
from torch import nn
from torch.nn import functional as F   
# transitionで使用されている
class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers

        self.conv2d = nn.Conv2d(in_channels=self._channels, out_channels=self._channels, kernel_size=3, stride=1, padding=1)

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                            out_channels=self._channels,
                                            strides=1,
                                            do_batch_norm=do_batch_norm) for i in range(self._layers)])

    def forward(self,input_res):
        inputs = input_res.clone()
        x = self.conv2d(inputs)
        x = F.layer_norm(x, [inputs.shape[2],inputs.shape[3]])
        x = F.relu(x)
        x = F.dropout(x, p=0.0)

        input = inputs + x
        return input

# decorderで使用されている
class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3,  do_batch_norm=False, dropout=0):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm
        self._dropout = dropout

        # conv2d, layer_norm, relu, dropout
        self.conv2d = nn.Conv2d(in_channels=self._in_channels, out_channels=self._out_channels, kernel_size=self._ksize, stride=2, padding=1)

        self.pad = nn.ReflectionPad2d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                        int((self._ksize-1)/2), int((self._ksize-1)/2)))

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')
        conv = self.pad(conv)

        # conv2d, layer_norm, relu, dropout
        conv = self.conv2d(conv)
        conv = F.layer_norm(conv, [conv.shape[2],conv.shape[3]])
        conv = F.relu(conv)
        conv = F.dropout(conv, p=self._dropout)

        # conv2d, layer_norm, tanh, dropout
        flow = self.conv2d(conv)
        flow = F.layer_norm(flow, [flow.shape[2],flow.shape[3]])
        flow = F.tanh(flow)
        flow = F.dropout(flow, p=self._dropout)
        flow = flow * 256

        return torch.cat([conv,flow.clone()], dim=1), flow

# encoder,decorderで使用されている
def general_conv2d(
        in_channels,
        out_channels, 
        ksize=3, 
        strides=2, 
        padding=1, 
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
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                F.layer_norm([out_channels,480 , 640]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                F.LayerNorm([out_channels, 480, 640]),
                nn.Tanh(),
                nn.Dropout(p=dropout)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh(),
                nn.Dropout(p=dropout)
            )
    return conv2d
