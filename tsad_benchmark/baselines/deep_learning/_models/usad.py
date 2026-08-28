
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_size: int, latent_size: int):
        super().__init__()
        hidden_1 = max(1, int(in_size / 2))
        hidden_2 = max(1, int(in_size / 4))
        self.linear1 = nn.Linear(in_size, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.linear3 = nn.Linear(hidden_2, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return self.relu(out)


class Decoder(nn.Module):
    def __init__(self, latent_size: int, out_size: int):
        super().__init__()
        hidden_1 = max(1, int(out_size / 4))
        hidden_2 = max(1, int(out_size / 2))
        self.linear1 = nn.Linear(latent_size, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.linear3 = nn.Linear(hidden_2, out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return self.sigmoid(out)


class USADNetwork(nn.Module):
    def __init__(self, w_size: int, z_size: int):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch, n: int):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = (1 / n) * torch.mean((batch - w1) ** 2) + (
            1 - 1 / n
        ) * torch.mean((batch - w3) ** 2)
        loss2 = (1 / n) * torch.mean((batch - w2) ** 2) - (
            1 - 1 / n
        ) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def validation_step(self, batch, n: int):
        with torch.no_grad():
            loss1, loss2 = self.training_step(batch, n)
        return {"val_loss1": loss1, "val_loss2": loss2}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses1 = [x["val_loss1"] for x in outputs]
        batch_losses2 = [x["val_loss2"] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {"val_loss1": epoch_loss1.item(), "val_loss2": epoch_loss2.item()}

    def window_scores(self, batch, alpha: float = 0.5, beta: float = 0.5):
        with torch.no_grad():
            w1 = self.decoder1(self.encoder(batch))
            w2 = self.decoder2(self.encoder(w1))
            return alpha * torch.mean((batch - w1) ** 2, dim=1) + beta * torch.mean(
                (batch - w2) ** 2, dim=1
            )


def evaluate(model: USADNetwork, val_loader, n: int, device):
    outputs = []
    for (batch,) in val_loader:
        outputs.append(model.validation_step(batch.to(device), n))
    return model.validation_epoch_end(outputs)


__all__ = ["Decoder", "Encoder", "USADNetwork", "evaluate"]
