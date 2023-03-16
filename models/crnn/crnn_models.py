import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, n_out]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, img_h, nc, n_class, nh, n_rnn=2, leakyRelu=False, lstmFlag=True):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'img_h has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            n_in = nc if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(n_out))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, n_class))
        else:
            self.linear = nn.Linear(nh * 2, n_class)

    def forward(self, x):
        # conv features
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
            # rnn features
            output = self.rnn(conv)
        else:
            T, b, h = conv.size()

            t_rec = conv.contiguous().view(T * b, h)

            output = self.linear(t_rec)  # [T * b, n_out]
            output = output.view(T, b, -1)
        output = output.permute(1, 0, 2).contiguous()

        return output
