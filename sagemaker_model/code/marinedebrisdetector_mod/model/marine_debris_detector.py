import torch
from torch import nn
from tqdm import tqdm


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

            # re-normalize scores to be at 0.5
            # y_score = normalize(y_score, self.model)

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


def normalize(score, model):
    return score * 0.5 / model.threshold


def plot_qualitative(detector):
    import numpy as np

    download_qualitative()

    X = np.load("qualitative_test.npz")["X"]
    Y = np.load("qualitative_test.npz")["Y"]
    ids = np.load("qualitative_test.npz")["ids"]

    X = torch.from_numpy(X).float()

    with torch.no_grad():
        y_pred, y_score = detector(X)

    import matplotlib.pyplot as plt
    from marinedebrisdetector.visualization import fdi, rgb

    N = X.shape[0]
    fig, axs = plt.subplots(N, 5, figsize=(5 * 3, N * 3))

    for ax, title in zip(axs[0], ["rgb", "fdi", "mask", "y_pred", "y_score"]):
        ax.set_title(title)

    for x, y, y_score_, y_pred_, id, ax_row in zip(
        X, Y, y_score.cpu(), y_pred.cpu(), ids, axs
    ):
        ax_row[0].imshow(rgb(x.numpy()).transpose(1, 2, 0))
        ax_row[1].imshow(fdi(x.numpy()))
        ax_row[2].imshow(y)
        ax_row[3].imshow(y_score_.squeeze().numpy())
        ax_row[4].imshow(y_pred_.squeeze().numpy())

        ax_row[0].set_ylabel(id)

        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def download(url):
    import os
    import urllib

    if not os.path.exists(os.path.basename(url)):
        output_path = os.path.basename(url)
        print(f"downloading {url} to {output_path}")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )
    else:
        print(f"{os.path.basename(url)} exists. skipping...")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_qualitative():
    download(
        "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/qualitative_test.npz"
    )


def download_accra():
    download(
        "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/accra_20181031.tif"
    )


def download_durban():
    download(
        "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/durban_20190424.tif"
    )
