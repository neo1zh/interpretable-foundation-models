import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import argparse

from model import SAE, ViTClassify, SAETrainer
from utils import (
    load_dataset_by_task, extract_embeddings, extract_augmented_activations, create_data_loaders, 
    set_seed, save_activations, load_activations, image_augment,
    init_wandb, log_wandb_metrics, log_wandb_model_artifacts, 
    log_wandb_activations, log_wandb_histogram, finish_wandb
)
from evaluate import compute_all_metrics, print_metrics
from config import parser


def train_sae(args):
    """Train SAE model"""
    print("Starting SAE training...")
    
    # Initialize wandb
    wandb_run = init_wandb(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    args.save_path = os.path.normpath(os.path.join(args.save_path, "SAE", args.task))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        print(f"✅ Created directory: {args.save_path}")
    
    # Load dataset
    print(f"Loading dataset: {args.task}")
    train_dataset, test_dataset, out_dim = load_dataset_by_task(args.task, args.data_path)
    args.out_dim = out_dim
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, 
        args.train_batch_size, args.eval_batch_size, args.num_workers
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Initialize ViT model
    print("Initializing ViT model...")
    model = ViTClassify(
        out_dim=args.out_dim, 
        layer=args.layer,
    )
    model = model.cuda()
    
    # Load pretrained ViT weights if available
    if args.use_finetuned_vit and os.path.exists(args.use_finetuned_vit):
        model_path = os.path.join(args.load_path, 'ViT-pretrain', f'{args.task}_epoch{args.pretrain_epoch}.pt')
        print(f"Loading pretrained ViT from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print("✅ Loaded pretrained ViT model")
    else:
        print("Using pretrained ViT model from HuggingFace")
    
    # Set SAE activation dimension
    if args.d_activation is None:
        args.d_activation = args.d_embedding * args.expansion_factor
    
    print(f"SAE activation dimension: {args.d_activation}")
    
    # Initialize SAE model
    print("Initializing SAE model...")
    sae_model = SAE(
        d_embedding=args.d_embedding,
        d_activation=args.d_activation,
        expansion_factor=args.expansion_factor,
        sparsity_penalty=args.sparsity_penalty,
    )
    
    # Load pretrained SAE if specified
    if args.load_pretrained_sae and os.path.exists(args.load_pretrained_sae):
        print(f"Loading pretrained SAE from: {args.load_pretrained_sae}")
        sae_model.load_state_dict(torch.load(args.load_pretrained_sae, map_location=args.device))
        sae_model = sae_model.to(args.device)  # Move model to device after loading
        print("✅ Loaded pretrained SAE model")
    elif args.load_pretrained_sae:
        print(f"⚠️  Pretrained SAE path not found: {args.load_pretrained_sae}")
        print("Starting training from scratch...")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        sae_model.parameters(), 
        lr=args.sae_lr, 
        weight_decay=args.sae_weight_decay
    )
    
    # Initialize SAE trainer
    sae_trainer = SAETrainer(sae_model, optimizer, device=args.device)
    
    # Extract embeddings for SAE training
    print("Extracting ViT embeddings...")
    train_embeddings, train_labels, train_preds = extract_embeddings(model, train_loader, args.device, args.layer, args.use_cls_token_only)
    test_embeddings, test_labels, test_preds = extract_embeddings(model, test_loader, args.device, args.layer, args.use_cls_token_only)
    
    # Extract augmented embeddings for stability evaluation (if enabled)
    train_embeddings_aug = None
    if args.use_augmentation:
        print("Extracting augmented ViT embeddings for stability evaluation...")
        train_embeddings_aug= extract_augmented_activations(
            model, train_loader, args.device, args.layer, args.augmentation_strength
        )
        # train_embeddings_aug = torch.cat(train_embeddings_aug, dim=0)
        print(f"Augmented train embeddings shape: {train_embeddings_aug.shape}")
    
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    # Create embedding data loaders
    train_embedding_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
    test_embedding_dataset = torch.utils.data.TensorDataset(test_embeddings, test_labels)
    
    train_embedding_loader = torch.utils.data.DataLoader(
        train_embedding_dataset, 
        batch_size=args.sae_batch_size, 
        shuffle=True
    )
    test_embedding_loader = torch.utils.data.DataLoader(
        test_embedding_dataset, 
        batch_size=args.sae_batch_size, 
        shuffle=False
    )
    
    # Training loop
    print("Starting SAE training...")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.sae_epochs):
        # Training
        sae_model.train()
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        
        for batch_embeddings, batch_labels in tqdm(train_embedding_loader, desc=f"Epoch {epoch+1}/{args.sae_epochs}"):
            batch_embeddings = batch_embeddings.to(args.device)
            
            loss_dict = sae_trainer.train_step(batch_embeddings)
            
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['reconstruction_loss']
            total_sparsity_loss += loss_dict['sparsity_loss']
        
        # Average losses
        avg_loss = total_loss / len(train_embedding_loader)
        avg_recon_loss = total_recon_loss / len(train_embedding_loader)
        avg_sparsity_loss = total_sparsity_loss / len(train_embedding_loader)
        
        # Log training metrics to wandb
        log_wandb_metrics({
            'train/total_loss': avg_loss,
            'train/reconstruction_loss': avg_recon_loss,
            'train/sparsity_loss': avg_sparsity_loss,
            'train/epoch': epoch + 1
        }, step=epoch)
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            sae_model.eval()
            eval_loss = 0
            all_activations = []
            all_labels = []
            
            with torch.no_grad():
                for batch_embeddings, batch_labels in test_embedding_loader:
                    batch_embeddings = batch_embeddings.to(args.device)
                    
                    loss_dict, activations = sae_trainer.evaluate(batch_embeddings)
                    eval_loss += loss_dict['total_loss']
                    
                    all_activations.append(activations.cpu())
                    all_labels.append(batch_labels)
            
            eval_loss /= len(test_embedding_loader)
            all_activations = torch.cat(all_activations, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            
            print(f"Epoch {epoch+1}/{args.sae_epochs}")
            print(f"Train Loss: {avg_loss:.6f} (Recon: {avg_recon_loss:.6f}, Sparsity: {avg_sparsity_loss:.6f})")
            print(f"Eval Loss: {eval_loss:.6f}")
            
            # Log evaluation metrics to wandb
            log_wandb_metrics({
                'eval/total_loss': eval_loss,
                'eval/epoch': epoch + 1
            }, step=epoch)
            
            # Log activation histograms to wandb
            log_wandb_histogram(all_activations.flatten(), 'activations/distribution', step=epoch)
            log_wandb_histogram(np.mean(all_activations, axis=0), 'activations/mean_per_feature', step=epoch)
            
            # Early stopping
            if eval_loss < best_loss:
                best_loss = eval_loss
                patience_counter = 0
                
                # Save best model
                torch.save(sae_model.state_dict(), os.path.join(args.save_path, 'best_sae_model.pt'))
                
                # Log best model to wandb
                log_wandb_model_artifacts(sae_model, args.save_path, epoch=epoch+1, is_best=True)
                
                # Save activations
                if args.save_activations:
                    save_activations(all_activations, all_labels, args.save_path, epoch+1)
                    # Log activations to wandb
                    log_wandb_activations(all_activations, all_labels, args.save_path, epoch+1)
            else:
                patience_counter += 1
            
            if patience_counter >= args.sae_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Logging
        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}")
    
    print("SAE training completed!")
    
    # Load best model
    best_model_path = os.path.join(args.save_path, 'best_sae_model.pt')
    if os.path.exists(best_model_path):
        sae_model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        sae_model = sae_model.to(args.device)  # Move model to device after loading
        print("Loaded best SAE model")
    
    # Log final model to wandb
    log_wandb_model_artifacts(sae_model, args.save_path, is_best=True)
    
    return sae_model, train_embeddings, train_labels, train_preds, test_embeddings, test_labels, test_preds, train_embeddings_aug


def evaluate_sae(args, sae_model, train_embeddings, train_preds, test_embeddings, test_preds, train_embeddings_aug=None):
    """Evaluate SAE model with memory-efficient batch processing"""
    print("Evaluating SAE model...")
    
    # Move SAE model to device
    sae_model = sae_model.to(args.device)
    sae_model.eval()
    
    # Process activations in batches to avoid memory issues
    batch_size = args.sae_batch_size
    
    print("Computing train activations...")
    train_activations = compute_activations_batch(sae_model, train_embeddings, batch_size, args.device)
    
    print("Computing test activations...")
    test_activations = compute_activations_batch(sae_model, test_embeddings, batch_size, args.device)
    
    train_activations_aug = None
    if train_embeddings_aug is not None:
        print("Computing augmented train activations...")
        train_activations_aug = compute_activations_batch(sae_model, train_embeddings_aug, batch_size, args.device)
    
    # Move activations to CPU for sklearn processing
    train_activations = train_activations.cpu()
    test_activations = test_activations.cpu()
    if train_activations_aug is not None:
        train_activations_aug = train_activations_aug.cpu()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print("Computing metrics...")
    # Compute metrics with stability evaluation if augmented activations available
    metrics = compute_all_metrics(
        train_activations, train_preds,
        test_activations, test_preds,
        concept_orig=train_activations,
        concept_aug=train_activations_aug if train_activations_aug is not None else None
    )
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics
    metrics_path = os.path.join(args.save_path, 'sae_metrics.npy')
    np.save(metrics_path, metrics)
    print(f"Metrics saved to: {metrics_path}")
    
    # Log evaluation metrics to wandb
    log_wandb_metrics(metrics, prefix="final_eval")
    
    return metrics


def compute_activations_batch(sae_model, embeddings, batch_size, device):
    """Compute SAE activations in batches to manage memory"""
    activations_list = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i+batch_size].to(device)
            _, batch_activations = sae_model(batch_embeddings)
            activations_list.append(batch_activations.cpu())
            
            # Clear intermediate tensors
            del batch_embeddings, batch_activations
    
    return torch.cat(activations_list, dim=0)


def main():
    args = parser.parse_args()
    
    if args.train:
        # Train SAE
        sae_model, train_embeddings, train_labels, train_preds, test_embeddings, test_labels, test_preds, train_embeddings_aug = train_sae(args)
        
        # Evaluate SAE
        metrics = evaluate_sae(args, sae_model, train_embeddings, train_preds, test_embeddings, test_preds, train_embeddings_aug)
        
        # Finish wandb run
        finish_wandb()
    else:
        print("Evaluation only mode - loading pretrained SAE...")
        
        # Initialize wandb for evaluation only
        wandb_run = init_wandb(args)
        
        # Load pretrained SAE model
        sae_model = SAE(
            d_embedding=args.d_embedding,
            d_activation=args.d_activation if args.d_activation else args.d_embedding * args.expansion_factor,
            expansion_factor=args.expansion_factor,
            sparsity_penalty=args.sparsity_penalty,
        )
        
        model_path = os.path.join(args.save_path, 'best_sae_model.pt')
        if os.path.exists(model_path):
            sae_model.load_state_dict(torch.load(model_path, map_location=args.device))
            sae_model = sae_model.to(args.device)  # Move model to device after loading
            print("Loaded pretrained SAE model")
        else:
            print(f"SAE model not found at: {model_path}")
            finish_wandb()
            return
        
        # Load embeddings and labels
        if args.load_activations:
            train_activations, train_labels = load_activations(args.save_path, args.sae_epochs)
            test_activations, test_labels = load_activations(args.save_path, args.sae_epochs)
        else:
            # Extract embeddings
            set_seed(args.seed)
            train_dataset, test_dataset, out_dim = load_dataset_by_task(args.task, args.data_path)
            train_loader, test_loader = create_data_loaders(
                train_dataset, test_dataset, 
                args.train_batch_size, args.eval_batch_size, args.num_workers
            )
            
            model = ViTClassify(
                out_dim=out_dim, 
                layer=args.layer,
            )
            model = model.cuda()
            
            if not args.use_finetuned_vit:
                pretrain_path = os.path.join(args.load_path, 'ViT-pretrain', f'{args.task}_epoch{args.pretrain_epoch}.pt')
                if os.path.exists(pretrain_path):
                    model.load_state_dict(torch.load(pretrain_path))
            
            train_embeddings, train_labels, train_preds = extract_embeddings(model, train_loader, args.device, args.layer, args.use_cls_token_only)
            test_embeddings, test_labels, test_preds = extract_embeddings(model, test_loader, args.device, args.layer, args.use_cls_token_only)
        
        # Evaluate SAE
        metrics = evaluate_sae(args, sae_model, train_embeddings, train_preds, test_embeddings, test_preds, train_embeddings_aug)
        
        # Finish wandb run
        finish_wandb()


if __name__ == "__main__":
    main()
