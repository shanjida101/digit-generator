import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=10):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, nz)
        self.model = nn.Sequential(
            nn.Linear(nz * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)
