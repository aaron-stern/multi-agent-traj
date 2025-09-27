import copy
import torch
from model.transformer import TrajectoryModel
import pytest


@pytest.fixture
def positions():
    return torch.randn(2, 10, 50, 2)


@pytest.fixture
def tm():
    return TrajectoryModel().eval()


def test_batching(positions, tm):
	perturbed = positions.clone()  # NOW it's a separate tensor
	perturbed[-1, ...] += 0.01
	assert not torch.allclose(tm(positions), tm(perturbed))
	assert torch.allclose(tm(positions)[0, ...], tm(perturbed)[0, ...])


def test_causal_mask(positions, tm):
	tm2 = copy.deepcopy(tm).eval()
	with torch.no_grad():
		tm2.tgt_seeds[-20, :] += 0.01
	assert torch.allclose(tm(positions)[0, :, 0, :], tm2(positions)[0, :, 0, :])
	assert not torch.allclose(tm(positions)[0, :, -1, :], tm2(positions)[0, :, -1, :])


def test_multi_agent_self_attention(positions, tm):
	tm2 = copy.deepcopy(tm).eval()
	with torch.no_grad():
		tm2.tgt_seeds[-1, :] += 0.01
	assert torch.allclose(tm(positions)[0, :, 0, :], tm2(positions)[0, :, 0, :])
	assert not torch.allclose(tm(positions)[0, :, -2, :], tm2(positions)[0, :, -2, :])
