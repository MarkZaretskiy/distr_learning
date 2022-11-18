import sys
sys.path.insert(0, '../')

import numpy as np

import torch
import dc_framework


def train_simple_model_cpu(model, criterion, data):

    model_dc_framework = dc_framework.init(model, criterion)
    model_dc_framework.train(train_data=data)
    model_dc_framework.save("tmp.pt")

def train_simple_model_gpu(model, criterion, data):

    model_dc_framework = dc_framework.init(model, criterion)
    model_dc_framework.cuda()
    model_dc_framework.train(train_data=data)
    model_dc_framework.save("tmp.pt")

def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    criterion = torch.nn.BCELoss()
    data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }

    train_simple_model_cpu(model, criterion, data)
    train_simple_model_gpu(model, criterion, data)


if __name__ == "__main__":

    main()
