# %%
run_name = "new"

import torch.utils.tensorboard
from collections import OrderedDict
from natsort import natsorted
import pathlib
import random
import shutil
from itertools import groupby

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


import lightning.pytorch.utilities.seed as seed
import torch
from torchvision.datasets import ImageFolder
import lightning.pytorch as pl
import lightning.pytorch.loggers
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image

import torch.utils.data as tdata
import torchmetrics
import torchmetrics.classification
import torchvision
import pydantic
from torch.utils.data import Subset, ConcatDataset

from more_itertools import partition

torch.set_float32_matmul_precision("high")


dataset_name = "n11939491"


def const_init(model, fill=0.0):
    for name, param in model.named_parameters():
        param.data.fill_(fill)


def shuffled(x):
    x = list(x)
    return random.sample(x, len(x))


segments_dir = pathlib.Path("./data/segments/")
train_dir = pathlib.Path("./data/train/")

save_dir = pathlib.Path("./models/") / run_name
save_dir.mkdir(exist_ok=True, parents=True)
(save_dir / "best").mkdir(exist_ok=True, parents=True)


class Tree(pydantic.BaseModel):
    tree: dict[str, list[str]]
    start_dir: pathlib.Path
    end_dir: pathlib.Path
    save_dir: pathlib.Path


tree = Tree(
    tree={
        "petal": [],
        "disk": [],
        "flower_head": ["disk", "petal"],
        "leaf": [],
        "stem": [],
        dataset_name: ["flower_head", "stem", "leaf"],
    },
    start_dir=segments_dir,
    end_dir=train_dir,
    save_dir=save_dir,
)

all_cnns = {}


def load_frozen(model, ckpt, _req_grad=False, **kwds):
    result = model.load_from_checkpoint(ckpt, **kwds).to("cuda")
    result.eval()
    for i in result.parameters():
        i.requires_grad = _req_grad
    return result


class TreeCNN(pl.LightningModule):
    cnn_children: dict[str, "TreeCNN"]
    name: str

    class Constants(pydantic.BaseModel):
        data_dir: pathlib.Path
        batch_size: int
        crop_size: int
        learning_rate: float

    constants: Constants

    class Dataset(pydantic.BaseModel):
        train: tdata.Dataset | None
        val: tdata.Dataset | None
        test: tdata.Dataset | None

        all_test: tdata.Dataset | None

        class Config:
            arbitrary_types_allowed = True

    dataset: Dataset

    def __init__(
        self,
        tree: Tree,
        name: str,
        is_root: bool = False,
        inference: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        best = ""

        self.is_root = is_root
        self.name = name

        self.cnn_children = {
            children_name: all_cnns.setdefault(
                children_name,
                load_frozen(
                    model=self.__class__,
                    ckpt=tree.save_dir / best / f"{children_name}.ckpt",
                    tree=tree,
                    name=children_name,
                    inference=inference,
                ),
            )
            for children_name in tree.tree[name]
        }

        is_leaf = not self.cnn_children

        def get_constants(is_leaf: bool, is_root: bool) -> "TreeCNN.Constants":
            return TreeCNN.Constants(
                learning_rate=5e-3,
                batch_size=256,
                data_dir=tree.end_dir if is_root else tree.start_dir,
                crop_size=256 if is_leaf and not is_root else 512,
            )

        self.constants = get_constants(is_leaf=is_leaf, is_root=is_root)

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.classification.BinaryF1Score()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        self.dataset_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
                torchvision.transforms.CenterCrop(
                    (self.constants.crop_size, self.constants.crop_size)
                ),
                self.transform,
            ]
        )

        def get_model(is_leaf):
            def factory_conv2d(
                input_channels,
                output_channels,
                kernel_size,
                pool_size=4,
                batch_norm=True,
            ):
                return nn.Sequential(
                    OrderedDict(
                        {
                            "conv1": nn.Conv2d(
                                input_channels, output_channels, kernel_size, bias=False
                            ),
                            "norm": nn.BatchNorm2d(output_channels)
                            if batch_norm
                            else nn.Identity(),
                            "relu1": nn.ReLU(),
                            "pool": nn.MaxPool2d(pool_size)
                            if pool_size
                            else nn.Identity(),
                        }
                    )
                )

            def factory_linear(
                input_size,
                output_size,
                activation: type[nn.Module] = nn.ReLU,
                activation_kwargs=dict(),
            ):
                return nn.Sequential(
                    OrderedDict(
                        {
                            "flatten": nn.Flatten(),
                            "lin": nn.Linear(input_size, output_size),
                            "activation": activation(**activation_kwargs),
                        }
                    )
                )

            return nn.Sequential(
                OrderedDict(
                    {
                        "cnn": nn.Sequential(
                            OrderedDict(
                                {
                                    "conv2d_1": factory_conv2d(3, 8, 5),
                                    "conv2d_2": factory_conv2d(8, 4, 3),
                                    "conv2d_3": factory_conv2d(4, 1, 3),
                                }
                            )
                            if is_leaf
                            else OrderedDict(
                                {
                                    "conv2d_1": factory_conv2d(
                                        len(self.cnn_children), 1, 3, pool_size=2
                                    ),
                                }
                            )
                        ),
                        "inference": nn.Sequential(
                            OrderedDict(
                                {
                                    "lin": factory_linear(
                                        9 if is_leaf else 4, 1, nn.Sigmoid
                                    ),
                                }
                            )
                        ),
                    }
                )
            )

        self.model = get_model(is_leaf)

    ### DATA
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        data_dir = self.constants.data_dir
        train, val, test, all_test = None, None, None, None
        dir_index = list(sorted([i.name for i in data_dir.iterdir()])).index(self.name)

        def balance_dataset(dataset, normal=True):
            if normal:
                correct_ix, incorrect_ix = partition(
                    lambda i: dataset.target_transform(i[1]),
                    [(num, trg) for num, (_, trg) in enumerate(dataset.samples)],
                )
                correct_ix, incorrect_ix = (
                    shuffled([i[0] for i in correct_ix]),
                    shuffled([i[0] for i in incorrect_ix]),
                )

                min_samples = min(len(correct_ix), len(incorrect_ix))

                subset_correct = Subset(dataset, correct_ix[:min_samples])
                subset_incorrect = Subset(dataset, incorrect_ix[:min_samples])

                return ConcatDataset([subset_correct, subset_incorrect])

            else:
                groups = {
                    i[0]: [x[0] for x in i[1]]
                    for i in groupby(enumerate(dataset.samples), lambda i: i[1][1])
                }
                correct_ix = groups[dir_index]
                incorrect_ix = [
                    dataset[0] for ix, dataset in groups.items() if ix != dir_index
                ]
                subset_correct = Subset(dataset, correct_ix)
                subset_incorrect = Subset(dataset, incorrect_ix)

                return ConcatDataset([subset_correct, subset_incorrect])

        if stage == "fit" or stage == "validate" or stage is None:
            full_folder = ImageFolder(
                str(data_dir),
                transform=self.dataset_transform,
                target_transform=lambda x: torch.tensor(
                    x == dir_index, dtype=torch.float
                ),
            )

            balanced_dataset = balance_dataset(full_folder, normal=not self.is_root)

            train, val = tdata.random_split(balanced_dataset, [0.8, 0.2])

        if stage == "test":  # or stage is None:
            all_test = ImageFolder(
                str((all_classes_dir / ".." / "val").resolve()),
                transform=self.dataset_transform,
                target_transform=lambda x: torch.tensor(
                    x == dir_index, dtype=torch.float
                ),
            )
            test = balance_dataset(all_test)

        self.dataset = TreeCNN.Dataset(
            train=train, val=val, test=all_test, all_test=all_test
        )

    def dataloader(self, type, **kwargs):
        return tdata.DataLoader(
            getattr(self.dataset, type),
            batch_size=self.constants.batch_size,
            num_workers=24,
            **kwargs,
        )

    def train_dataloader(self):
        return self.dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")

    ### TRAINING
    def forward(self, x, transform=False, cnn=True):
        x = self.transform(x).to("cuda").unsqueeze(0) if transform else x

        with torch.no_grad():
            if not self.cnn_children:
                features = x
            else:
                stack = [
                    torchvision.transforms.functional.center_crop(
                        cnn.forward(x, cnn=True, transform=False), (6, 6)
                    )
                    for cnn in self.cnn_children.values()
                ]

                features = (
                    torch.stack(
                        stack,
                        dim=1,
                    ).squeeze(2)
                    if self.cnn_children
                    else x
                )

        return self.model[: (-1 if cnn else None)](features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, transform=False, cnn=False)
        loss = F.binary_cross_entropy(logits, y.unsqueeze(1))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, transform=False, cnn=False)

        loss = F.binary_cross_entropy(logits, y.unsqueeze(1))
        preds = logits
        self.accuracy(preds, y.unsqueeze(1))
        self.f1(preds, y.unsqueeze(1))

        self.log("val_f1", self.f1)

        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy)
        return {"loss": loss, "log": {"val_loss": loss, "val_acc": self.accuracy}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.constants.learning_rate)

    ### TES
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


all = ["disk", "petal", "flower_head", "leaf", "stem", "whole_flower", dataset_name]
toskip = set(all)

log_dir = pathlib.Path("lightning_logs") / "lightning_logs" / run_name
log_dir.mkdir(exist_ok=True, parents=True)
(save_dir / "train.py").write_text(pathlib.Path(__file__).read_text())

for i in (log_dir).iterdir():
    if i.name not in toskip:
        shutil.rmtree(i)


def train(model, save_path, stopping_threshold=0.95):
    model.setup()
    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs", version=f"{run_name}/{totrain}"
    )
    images, label = next(iter(model.train_dataloader()))
    logger.experiment.add_image("training_example", torchvision.utils.make_grid(images))

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        every_n_epochs=20,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=20,
        min_delta=1e-4,
        stopping_threshold=stopping_threshold,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        logger=logger,
        # deterministic=True
        callbacks=[model_checkpoint, early_stopping],
    )

    trainer.fit(model)

    trainer.save_checkpoint(save_path)

    save_best_path = save_path.parent / "best" / f"{totrain}.ckpt"
    save_best_path.unlink(missing_ok=True)
    save_best_path.symlink_to(pathlib.Path(model_checkpoint.best_model_path).resolve())


pl.seed_everything(
    1822,
)

for ix, (totrain, children) in enumerate(tree.tree.items()):
    if totrain in toskip:
        continue

    node = TreeCNN(
        tree=tree,
        name=totrain,
        is_root=ix == len(tree.tree.items()) - 1,
        inference=False,
    ).to("cuda")

    train(
        node,
        save_path=save_dir / f"{totrain}.ckpt",
        stopping_threshold=0.85 if children else 0.9,
    )


n119 = load_frozen(
    model=TreeCNN,
    ckpt=save_dir / "best" / f"{dataset_name}.ckpt",
    tree=tree,
    name=dataset_name,
    is_root=True,
    inference=False,
)
trainer = pl.Trainer()
print(trainer.test(n119))
