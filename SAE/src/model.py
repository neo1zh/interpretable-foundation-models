import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTForImageClassification
import numpy as np


class SAE(nn.Module):
    """
    Sparse Autoencoder for ViT activations
    """
    def __init__(self, d_embedding=768, d_activation=None, expansion_factor=4, sparsity_penalty=1e-3, use_all_layers=False):
        super(SAE, self).__init__()
        
        # SAE dimensions
        self.d_embedding = d_embedding
        self.d_activation = d_activation if d_activation is not None else d_embedding * expansion_factor
        self.sparsity_penalty = sparsity_penalty
        self.use_all_layers = use_all_layers
        
        self.input_dim = d_embedding
        
        # Encoder: maps from embedding space to activation space
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.d_activation),
            nn.ReLU()
        )
        
        # Decoder: maps from activation space back to embedding space
        self.decoder = nn.Sequential(
            nn.Linear(self.d_activation, self.input_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through SAE
        Args:
            x: input embeddings of shape (batch_size, seq_len, d_embedding) or (batch_size, d_embedding)
        Returns:
            reconstructed: reconstructed embeddings
            activations: sparse activations
        """
        # Handle input based on use_all_layers flag and actual input shape
        if self.use_all_layers:
            # For use_all_layers=True, we expect input from all 12 layers concatenated
            if len(x.shape) == 3:
                # x shape: (batch_size, total_tokens, d_embedding)
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)  # Flatten to (batch_size, total_tokens * d_embedding)
            else:
                # Already flattened for all layers case
                x_flat = x
        else:
            # For single layer case, extract_embeddings already flattens to (batch_size * seq_len, d_embedding)
            if len(x.shape) == 3:
                # x shape: (batch_size, seq_len, d_embedding)
                batch_size, seq_len, d_emb = x.shape
                x_flat = x.view(batch_size, -1)  # Flatten to (batch_size, seq_len * d_embedding)
            else:
                # Already flattened to (batch_size * seq_len, d_embedding)
                x_flat = x
        
        # Encode to sparse activations
        activations = self.encoder(x_flat)
        
        # Decode back to embedding space
        reconstructed_flat = self.decoder(activations)
        
        # Reshape back to original shape
        if self.use_all_layers:
            # For all layers case, reshape to (batch_size, total_tokens, d_embedding)
            if len(x.shape) == 3:
                reconstructed = reconstructed_flat.view(batch_size, self.total_tokens, self.d_embedding)
            else:
                # Input was already flattened, need to infer batch_size
                batch_size = x.shape[0] // self.total_tokens
                reconstructed = reconstructed_flat.view(batch_size, self.total_tokens, self.d_embedding)
        else:
            # For single layer case, reshape back to original shape
            if len(x.shape) == 3:
                reconstructed = reconstructed_flat.view(batch_size, x.shape[1], x.shape[2])
            else:
                # Input was already flattened, keep flattened
                reconstructed = reconstructed_flat
        
        return reconstructed, activations
    
    def compute_loss(self, x, reconstructed, activations):
        """
        Compute SAE loss with sparsity penalty
        Args:
            x: original embeddings
            reconstructed: reconstructed embeddings
            activations: sparse activations
        Returns:
            total_loss: combined reconstruction and sparsity loss
            reconstruction_loss: MSE loss
            sparsity_loss: L1 penalty on activations
        """
        # Flatten inputs for loss computation if needed
        if len(x.shape) == 3:
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
            
        if len(reconstructed.shape) == 3:
            reconstructed_flat = reconstructed.view(reconstructed.shape[0], -1)
        else:
            reconstructed_flat = reconstructed
        
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed_flat, x_flat)
        
        # Sparsity loss (L1 penalty)
        sparsity_loss = torch.mean(torch.abs(activations))
        
        # Total loss
        total_loss = reconstruction_loss + self.sparsity_penalty * sparsity_loss
        
        return total_loss, reconstruction_loss, sparsity_loss


class ViTClassify(nn.Module):
    """
    ViT model wrapper for SAE training
    """
    def __init__(self, out_dim=5, layer=-2, 
                ):
        super(ViTClassify, self).__init__()
        self.out_dim = out_dim
        self.layer = layer

        # Load pretrained ViT
        ViT_config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224-in21k", 
            output_hidden_states=True, 
            output_attentions=True, 
            num_labels=out_dim
        )
        self.ViT = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k', 
            config=ViT_config
        )
        
        # Freeze ViT parameters for SAE training
        for param in self.ViT.parameters():
            param.requires_grad = False
            
        self.embedding = None
    

    def forward(self, encodings, labels=None):
        ViT_output = self.ViT(encodings)
        logits = ViT_output['logits']
        all_states = ViT_output['hidden_states']
        attention = ViT_output['attentions']
        states = all_states[self.layer]
        self.embedding = states

        return logits, states, attention


class SAETrainer:
    """
    Trainer class for SAE
    """
    def __init__(self, sae_model, optimizer, device='cuda'):
        self.sae_model = sae_model
        self.optimizer = optimizer
        self.device = device
        self.sae_model.to(device)
    
    def train_step(self, embeddings):
        """
        Single training step for SAE
        Args:
            embeddings: ViT embeddings of shape (batch_size, d_embedding)
        Returns:
            loss_dict: dictionary containing loss components
        """
        self.sae_model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        reconstructed, activations = self.sae_model(embeddings)
        
        # Compute loss
        total_loss, reconstruction_loss, sparsity_loss = self.sae_model.compute_loss(
            embeddings, reconstructed, activations
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }
    
    def evaluate(self, embeddings):
        """
        Evaluate SAE on given embeddings
        Args:
            embeddings: ViT embeddings of shape (batch_size, d_embedding)
        Returns:
            loss_dict: dictionary containing loss components
            activations: sparse activations
        """
        self.sae_model.eval()
        
        with torch.no_grad():
            reconstructed, activations = self.sae_model(embeddings)
            total_loss, reconstruction_loss, sparsity_loss = self.sae_model.compute_loss(
                embeddings, reconstructed, activations
            )
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }, activations
