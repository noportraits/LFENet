import torch.nn as nn
import torch
from modules.CSM import CSM


class CVMI(nn.Module):

    def __init__(self, channels):
        super(CVMI, self).__init__()
        self.fe = CSM(channels)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

    def forward(self, low1, low2):
        Q, K = self.conv(self.fe(low1)), self.conv(self.fe(low2))
        b, c, h, w = Q.shape
        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
        left = low1 + torch.bmm(self.relu(self.softmax(score)),
                                low2.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)
        right = low2 + torch.bmm(self.relu(self.softmax(score.permute(0, 2, 1))),
                                 low1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)

        return left, right