import torch
from torch import nn

class TrajectoryModel(nn.Module):
	def __init__(self, tgt_num_steps: int = 50, num_agents: int = 10, d_in: int = 2):
		super().__init__()
		nhead = 4
		d_model = 32
		self.input_projection = nn.Linear(d_in, d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

		self.agent_embeddings = nn.Embedding(num_agents, d_in)

		self.encoder = torch.nn.TransformerEncoder(
			encoder_layer, 
			num_layers=1, 
			norm=nn.LayerNorm(d_model),
			enable_nested_tensor=False, 
			mask_check=True,
		)
		decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
		self.decoder = nn.TransformerDecoder(
			decoder_layer=decoder_layer,
			num_layers=2,
			norm=nn.LayerNorm(d_model),
		)

		self.mlp = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, 2),
			nn.Tanh(),
		)
		
		self.tgt_num_steps = tgt_num_steps
		self.tgt_seeds = nn.Parameter(
			torch.randn((tgt_num_steps * num_agents, d_model))
		)
		self.causal_mask = self._create_causal_mask(tgt_num_steps, num_agents)

	
	@staticmethod
	def _create_causal_mask(tgt_num_steps: int, num_agents: int) -> torch.Tensor:
		# Create causal mask with staircase pattern
		mask_size = tgt_num_steps * num_agents
		causal_mask = torch.full((mask_size, mask_size), float('-inf'))
		for i in range(tgt_num_steps):
			for j in range(i + 1):
				start_row = i * num_agents
				end_row = (i + 1) * num_agents
				start_col = j * num_agents
				end_col = (j + 1) * num_agents
				causal_mask[start_row:end_row, start_col:end_col] = 0

		return causal_mask


	def flatten_data(self, x: torch.Tensor):
		"""x is shape (batch, num_agents, num_steps, d). 
		This function flattens it to (batch, num_agents * num_steps, d),
		in order like (batch1, agent1/step1, d), (batch1, agent1/step2, d), ...), etc.
		"""
		batch_size, num_agents, num_steps, d = x.shape
		return x.reshape(batch_size, num_agents * num_steps, d)


	def unflatten_data(self, x: torch.Tensor, num_agents: int, num_steps: int):
		"""x is shape (batch, num_agents * num_steps, d). 
		This function unflattens it to (batch, num_agents, num_steps, d),
		in order like (batch1, agent1/step1, d), (batch1, agent1/step2, d), ...), etc.
		"""
		batch_size, _, d = x.shape
		return x.reshape(batch_size, num_agents, num_steps, d)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		num_agents = x.shape[1]
		agent_embeddings = self.agent_embeddings(torch.arange(num_agents))
		x = x + agent_embeddings.reshape(1, num_agents, 1, -1)
		x = self.flatten_data(x)
		x = self.input_projection(x)
		encodings = self.encoder(x)
		tgt = self.tgt_seeds.unsqueeze(0).expand(x.size(0), -1, -1)
		decodings = self.decoder(
			tgt=tgt,
			memory=encodings,
			tgt_mask=self.causal_mask,
		)
		decoder_representations = self.unflatten_data(decodings, num_agents, self.tgt_num_steps)
		predictions = self.mlp(decoder_representations)
		return predictions * 200.0