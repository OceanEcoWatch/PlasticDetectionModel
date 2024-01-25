import torch
from torch import nn


class MarineDebrisDetector(nn.Module):
    def __init__(
        self,
        model: str = "unetpp",
        seed: int = 1,
        ensemble: bool = False,
        test_time_augmentation: bool = False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not ensemble:
            self.model = (
                torch.hub.load("marccoru/marinedebrisdetector", model, int(seed))
                .to(self.device)
                .eval()
            )
        if ensemble:
            self.model = [
                torch.hub.load("marccoru/marinedebrisdetector", model, seed)
                .to(self.device)
                .eval()
                for seed in [1, 2, 3]
            ]

        if test_time_augmentation:
            if not ensemble:
                self.model = TestTimeAugmentationWrapper(self.model)
            else:
                self.model = [
                    TestTimeAugmentationWrapper(model) for model in self.model
                ]

    def forward(self, X):
        if isinstance(self.model, list):  # ensemble
            y_score = [torch.sigmoid(model(X.to(self.device))) for model in self.model]

            # normalize scores to be at threshold 0.5
            y_pred = torch.median(
                torch.stack(
                    [y_sc > model.threshold for y_sc, model in zip(y_score, self.model)]
                ),
                dim=0,
            ).values

            return y_score, y_pred

        else:
            y_score = torch.sigmoid(self.model(X.to(self.device)))

            return y_score, y_score > self.model.threshold


class TestTimeAugmentationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threshold = model.threshold

    def forward(self, x):
        y_logits = self.model(x)

        y_logits += torch.fliplr(self.model(torch.fliplr(x)))  # fliplr)
        y_logits += torch.flipud(self.model(torch.flipud(x)))  # flipud

        for rot in [1, 2, 3]:  # 90, 180, 270 degrees
            y_logits += torch.rot90(
                self.model(torch.rot90(x, rot, [2, 3])), -rot, [2, 3]
            )
        y_logits /= 6

        return y_logits
