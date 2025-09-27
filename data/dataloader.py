from torch.utils.data import DataLoader
from data.dataset import TrajectoryDataset

def create_dataloader(dataset: TrajectoryDataset, batch_size: int, shuffle: bool = True):
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)