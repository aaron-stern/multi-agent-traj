import torch
from data.dataset import generate_dataset
from model.transformer import TrajectoryModel
from torch.utils.data import DataLoader


if __name__ == "__main__":

	history_steps = 10
	lookahead_steps = 10
	num_agents = 10
	num_samples = 256

	batch_size = 32
	num_epochs = 200
	
	dataset = generate_dataset(
		num_samples=num_samples, 
		num_balls=num_agents, 
		num_steps=history_steps + lookahead_steps,
		lookahead_steps=lookahead_steps,
	)
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

	item = train_dataset[0]
	d_in = item['history'].shape[-1]
	model = TrajectoryModel(tgt_num_steps=lookahead_steps, num_agents=num_agents, d_in=d_in)

	def loss_fn(predictions, target_positions):
		return torch.sum((predictions - target_positions)**2) / (num_agents * lookahead_steps * batch_size)
	
	def eval_corr(predictions, target_positions) -> torch.Tensor:
		# predictions: (batch, num_agents, lookahead_steps, 2)
		# target_positions: (batch, num_agents, lookahead_steps, 2)
		
		# Flatten along dimensions 1 and 3 (agents and features)
		preds_flat = predictions.view(predictions.shape[0], -1, predictions.shape[2])  # (batch, num_agents*2, lookahead_steps)
		targets_flat = target_positions.view(target_positions.shape[0], -1, target_positions.shape[2])  # (batch, num_agents*2, lookahead_steps)
		
		# Compute correlation coefficient for each batch and each agent/feature combination
		total_corr = 0.0
		for b in range(preds_flat.shape[0]):
			batch_corrs = 0.0
			n_af = preds_flat.shape[1]
			for af in range(n_af):  # agent/feature combinations
				pred_series = preds_flat[b, af, :]  # (lookahead_steps,)
				target_series = targets_flat[b, af, :]  # (lookahead_steps,)
				
				correlation = torch.corrcoef(torch.stack([pred_series, target_series], dim=0))[0, 1]
				batch_corrs += correlation / n_af
			total_corr += batch_corrs / preds_flat.shape[0]
		return total_corr


	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(num_epochs):
		# Training phase
		model.train()
		train_loss = 0.0
		
		for batch in train_dataloader:
			optimizer.zero_grad()
			
			# Forward pass
			predictions = model(batch['history'])
			target_positions = batch['lookahead']['positions']
			
			# Calculate loss
			loss = loss_fn(predictions, target_positions)
			
			# Backward pass
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
		
		# Validation phase
		model.eval()
		val_loss = 0.0
		
		with torch.no_grad():
			for batch in val_dataloader:
				predictions = model(batch['history'])
				target_positions = batch['lookahead']['positions']
				loss = loss_fn(predictions, target_positions)
				val_loss += loss.item()

				val_corr = eval_corr(predictions, target_positions)
				val_corr += val_corr.item()
		
		# Calculate average losses
		avg_train_loss = train_loss / len(train_dataloader)
		avg_val_loss = val_loss / len(val_dataloader)
		avg_val_corr = val_corr / len(val_dataloader)
		
		# Log losses
		print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Corr: {avg_val_corr:.3f}")
	
	import pdb; pdb.set_trace()