import torch
import torchvision
from torch import nn


class ThyroidClassificationModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.BatchNorm1d(2),
            nn.Softmax(dim=-1)
        )
        self._is_inception3 = type(base_model) == torchvision.models.inception.Inception3
        if self._is_inception3:
            self.classifier2 = nn.Sequential(
                nn.Linear(1000, 500),
                nn.BatchNorm1d(500),
                nn.ReLU(),
                nn.Linear(500, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, 2),
                nn.BatchNorm1d(2),
                nn.Softmax(dim=-1)
            )

    def forward(self, x, validate=False):
        output = self.base_model(x.float())
        if self._is_inception3 and not validate:
            return self.classifier(output[0]), self.classifier2(output[1])
        return self.classifier(output)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
