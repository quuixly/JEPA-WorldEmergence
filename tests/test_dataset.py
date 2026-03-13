from dataset.dataset import OthelloDataset
import pytest
import torch

@pytest.fixture
def data():
    return torch.vstack([torch.arange(0, 60), torch.arange(0, 60)])

def test_len(monkeypatch, data):
    monkeypatch.setattr("dataset.dataset.torch.load", lambda path: data)

    dataset = OthelloDataset()
    assert len(dataset) == 2

def test_data_and_target(monkeypatch, data):
    monkeypatch.setattr("dataset.dataset.torch.load", lambda path: data)

    dataset = OthelloDataset()
    target = torch.arange(1, 61)
    target[59] = 0
    x, y = dataset[0]

    assert torch.equal(y, target)