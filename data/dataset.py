import torch
import math
import numpy as np
from torch.utils.data import Dataset
from simulation.generate import generate_trajectory_data

def generate_dataset(num_samples: int, num_balls=10, num_steps=50, lookahead_steps=20, dt=0.1, box_size=200, max_velocity=50, radius=20):
	positions = np.empty((num_samples, num_balls, num_steps, 2))  # pyright: ignore[reportUndefinedVariable]
	velocities = np.empty((num_samples, num_balls, num_steps, 2))  # pyright: ignore[reportUndefinedVariable]
	for i in range(num_samples):
		positions_i, velocities_i = generate_trajectory_data(num_balls, num_steps, dt, box_size, max_velocity, radius)
		positions[i] = positions_i
		velocities[i] = velocities_i
	positions = torch.tensor(positions, dtype=torch.float32)
	velocities = torch.tensor(velocities, dtype=torch.float32)
	dataset = TrajectoryDataset(positions, velocities, lookahead_steps)
	return dataset

class TrajectoryDataset(Dataset):
	def __init__(
		self, 
		positions, 
		velocities,
		lookahead_steps,
	):
		self.positions = torch.tensor(positions)
		self.velocities = torch.tensor(velocities)
		self.positional_encodings = self.get_positional_encodings()
		self.lookahead_steps = lookahead_steps

	def get_positional_encodings(self, max_len: int = 200, d_model: int = 16) -> torch.Tensor:
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1).float()
		
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * 
			-(math.log(10000.0) / d_model)
		)
		
		pe[:, 0::2] = torch.sin(position * div_term)
		if d_model % 2 == 0:
			pe[:, 1::2] = torch.cos(position * div_term)
		else:
			pe[:, 1::2] = torch.cos(position * div_term[:-1])
		return pe[:self.positions.shape[2], :]

	def __len__(self):
		return len(self.positions)

	def __getitem__(self, idx):
		return {
			"history": torch.cat(
				[
					self.positions[idx, :, :-self.lookahead_steps, :], 
					self.positional_encodings[:-self.lookahead_steps, :].unsqueeze(0).expand(self.positions.shape[1], -1, -1),
				],
				dim=-1,
			),
			"lookahead": {
				"positions": self.positions[idx, :, -self.lookahead_steps:, :],
				"positional_encodings": self.positional_encodings[-self.lookahead_steps:, :]
			}
		}

