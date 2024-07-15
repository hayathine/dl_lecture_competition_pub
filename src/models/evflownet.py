import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any
import numpy as np

_BASE_CHANNELS = 64
HEIGHT = 480
WIDTH = 640

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(
                                    self._args.num_bins, 
                                    _BASE_CHANNELS ,
                                    kernel_size=3, 
                                    stride=2, 
                                    height=HEIGHT,
                                    width=WIDTH,
                                    padding=1,
                                    dropout=self._args.dropout
                                    )
        
        self.encoder2 = general_conv2d(
                                    _BASE_CHANNELS, 
                                    2*_BASE_CHANNELS, 
                                    kernel_size=3, 
                                    stride=2, 
                                    height=HEIGHT/2,
                                    width=WIDTH/2,
                                    padding=1,
                                    dropout=self._args.dropout)
        
        self.encoder3 = general_conv2d(
                                    2*_BASE_CHANNELS, 
                                    4*_BASE_CHANNELS, 
                                    kernel_size=3, 
                                    stride=2, 
                                    height=HEIGHT/4,
                                    width=WIDTH/4,
                                    padding=1,
                                    dropout=self._args.dropout)
        
        self.encoder4 = general_conv2d(
                                    4*_BASE_CHANNELS, 
                                    8*_BASE_CHANNELS, 
                                    kernel_size=3, 
                                    height=HEIGHT/8,
                                    width=WIDTH/8,
                                    stride=2, 
                                    padding=1,
                                    dropout=self._args.dropout)

        self.resnet_block = nn.Sequential(*[build_resnet_block(
                                            8*_BASE_CHANNELS, 
                                            do_batch_norm=not self._args.no_batch_norm,
                                            height=HEIGHT/16,
                                            width=WIDTH/16,
                                            dropout=self._args.dropout
                                            ) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, 
                        do_batch_norm=not self._args.no_batch_norm,
                        height=HEIGHT/8,
                        width=WIDTH/8)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, 
                        do_batch_norm=not self._args.no_batch_norm,
                        height=HEIGHT/4,
                        width=WIDTH/4)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, 
                        do_batch_norm=not self._args.no_batch_norm,
                        height=HEIGHT/2,
                        width=WIDTH/2)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), 
                        do_batch_norm=not self._args.no_batch_norm,
                        height=HEIGHT,
                        width=WIDTH)
        
        self.dropout = nn.Dropout(p=self._args.dropout)

    def forward(self, inputs: Dict[str, Any]) :
        # encoder
        skip_connections = {}
        # torch.size([8, 2, 480, 640])
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        # torch.size([8, 2, 240, 320])
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        # torch.size([8, 2, 120, 160])
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        # torch.size([8, 2, 60, 80])
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        # flow0_torch.Size([8, 2, 60, 80])
        # flow1_torch.Size([8, 2, 120, 160])
        # flow2_torch.Size([8, 2, 240, 320])
        # flow3_torch.Size([8, 2, 480, 640])
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()
        # 最後のflowだけを用いているflow_dictを活用する
        # 各flowのサイズを[8, 2, 480, 640]に変更する
        for key in flow_dict.keys():
            flow_dict[key] = F.interpolate(flow_dict[key], size=[480,640], mode='nearest')
        return flow_dict, flow
        

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)