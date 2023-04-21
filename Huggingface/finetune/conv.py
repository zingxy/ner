import torch
import torch.nn as nn


class Conv(nn.Module):

    def __init__(self, in_c=512, window=4, output_channel=512):
        # 如果想要自定义初始化， 可以改写这里
        super(Conv, self).__init__()

        self.input_channel = in_c
        self.out_channel = output_channel
        self.window = window
        self.extractor = nn.Conv1d(self.input_channel, self.out_channel, self.window, padding='same')

    def forward(self, inputs):
        inputs_ = torch.transpose(inputs, 1, 2)
        output_ = self.extractor(inputs_)
        output = torch.transpose(output_, 1, 2)
        return output


if __name__ == '__main__':
    c = Conv()
