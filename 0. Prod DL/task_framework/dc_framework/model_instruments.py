import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict, Union, List, Callable

from dc_framework.data_preparation import Dataset

logger = logging.getLogger("__name__")


def init(model: torch.nn.Module, criterion: torch.nn.Module, metrics: Dict[str, Callable] = None):
    return DCFramework(model, criterion, metrics)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, metrics: Dict[str, Callable] = None, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion
        if metrics:
            self.metrics = torch.nn.ModuleDict({name:func for name, func in metrics})
        else:
            self.metrics = torch.nn.ModuleDict({"mae":torch.nn.L1Loss()})

        #check if model is already on cuda
        self.on_cuda = False
        if next(model.parameters()).is_cuda:
            self.on_cuda = True

    def forward(self, feature, target):
        try:
            if self.on_cuda:
                feature = feature.cuda()
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            if self.on_cuda:
                output = output.cuda()
                target = target.cuda()

            loss = self.criterion(output, target)
            results = {}
            for name, metric in self.metrics.items():
                results[name] = metric(output, target)
            results['output'] = output
            results['loss']   = loss
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return results

    def train(self, train_data: Dict[str, np.array], val_data: Union[Dict[str, np.array], None] = None, batch_size: int = 1, epochs: int = 2):

        train_dataloader = self._prepare_loader(train_data, batch_size=batch_size)
        if val_data:
            val_dataloader = self._prepare_loader(val_data, batch_size=batch_size)

        for i in range(epochs):

            self._train_loop(train_dataloader)

            if val_data:
                self._val_loop(val_dataloader) #torch.no_grad is incorporated into the function

    @torch.no_grad()
    def val(self, val_data: Dict[str, np.array], batch_size: int = 1):
        val_dataloader = self._prepare_loader(val_data, batch_size=batch_size)
        self._val_loop(val_dataloader)

    def _train_loop(self, loader):
        for batch in loader:
            output = self.forward(*batch)
            loss = output["loss"]
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def _val_loop(self, loader):
        for batch in loader:
            output = self.forward(*batch)
            loss = output["loss"]


    def _prepare_loader(self, data: Dict[str, np.array], batch_size: int = 1):
        data = Dataset(data)
        loader = data.get_data_loader(batch_size=batch_size)
        return loader

    def to(self, device: Union[torch.device, str]):
        device = torch.device(device) #cpu or gpu
        self.model.to(device)
        self.metrics.to(device)
        self.criterion.to(device)

    def cpu(self):
        self.to("cpu")
        self.on_cuda = False

    def cuda(self):
        self.to("cuda")
        self.on_cuda = True

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.on_cuda:
            self.to("cuda")