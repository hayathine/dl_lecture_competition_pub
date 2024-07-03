import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any
import numpy as np

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(
            in_channels = 4, 
            out_channels=_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout,
            x_size=240,
            y_size=320,
            activation='relu')
        self.encoder2 = general_conv2d(
            in_channels = _BASE_CHANNELS, 
            out_channels=2*_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout,
            x_size=120,
            y_size=160,
            activation='relu'
            )
        self.encoder3 = general_conv2d(
            in_channels = 2*_BASE_CHANNELS, 
            out_channels=4*_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout,
            x_size=60,
            y_size=80,
            activation='relu'
            )
        self.encoder4 = general_conv2d(
            in_channels = 4*_BASE_CHANNELS, 
            out_channels=8*_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout,
            x_size=30,
            y_size=40,
            activation='relu'
            )

        self.resnet_block = nn.Sequential(*[build_resnet_block(
                                            8*_BASE_CHANNELS, 
                                            do_batch_norm=not self._args.no_batch_norm) for i in range(2)]
                                            )

        self.decoder1 = upsample_conv2d_and_predict_flow(
            in_channels=16*_BASE_CHANNELS,
            out_channels=4*_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            x_size=30,
            y_size=40,
            dropout=self._args.dropout
            )

        self.decoder2 = upsample_conv2d_and_predict_flow(
            in_channels=8*_BASE_CHANNELS+2,
            out_channels=2*_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout)

        self.decoder3 = upsample_conv2d_and_predict_flow(
            in_channels=4*_BASE_CHANNELS+2,
            out_channels=_BASE_CHANNELS, 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout
            )

        self.decoder4 = upsample_conv2d_and_predict_flow(
            in_channels=2*_BASE_CHANNELS+2,
            out_channels=int(_BASE_CHANNELS/2), 
            do_batch_norm=not self._args.no_batch_norm,
            dropout=self._args.dropout
            )
        
        self.dropout = nn.Dropout(p=self._args.dropout)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
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
        # shape0 = flow_dict['flow0'].shape
        # shape1 = flow_dict['flow1'].shape
        # shape2 = flow_dict['flow2'].shape
        # shape3 = flow_dict['flow3'].shape
        # print(f'flow0_{shape0}')
        # print(f'flow1_{shape1}')
        # print(f'flow2_{shape2}')
        # print(f'flow3_{shape3}')
        # print(flow_dict.values())
        # flow = np.mean(flow_dict.values)
        return flow
        

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