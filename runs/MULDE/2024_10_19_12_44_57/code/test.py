import torch
import os
from models import MLPs, ScoreOrLogDensityNetwork  # Import the model definitions

# Initialize the model architecture exactly like you did during training
model = ScoreOrLogDensityNetwork(MLPs(input_dim=0,  # Set appropriate dimensions and arguments
                                      units=[4096, 4096],        # Adjust as needed
                                      dropout=0.5,
                                      layernorm=True),
                                 score_network=False).to('cpu')  # Use 'cpu' or 'cuda' as per your setup
# Define where to save the final model weights
save_model_dir = '/home/divyansh/Desktop/IPR/MULDE'
os.makedirs(save_model_dir, exist_ok=True)

# Path to save the final model weights
model_save_path = os.path.join(save_model_dir, '2_final_model_weights.pth')

# Save the trained model's state_dict (final weights)
torch.save(model.state_dict(), model_save_path)
print(f'Final model weights saved to {model_save_path}')

