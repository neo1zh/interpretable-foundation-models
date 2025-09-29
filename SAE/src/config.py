import argparse

parser = argparse.ArgumentParser(description='SAE for ViT Interpretability')

# Data arguments
parser.add_argument('--data_path', type=str, help='path of dataset', 
                    default='../dataset')
parser.add_argument('--task', type=str, help='task name of dataset', 
                    default='flower102', choices=['flower102', 'cub2011', 'cars', 'Color'])
parser.add_argument('--save_path', type=str, help='path to save', 
                    default='../ckpt')
parser.add_argument('--load_path', type=str, help='path to load pretrained ViT model', 
                    default='../ckpt')

# SAE model arguments
parser.add_argument('--d_embedding', type=int, help='ViT embedding dimension', default=768)
parser.add_argument('--expansion_factor', type=int, help='SAE expansion factor', default=4)
parser.add_argument('--d_activation', type=int, help='SAE activation dimension', default=None)
parser.add_argument('--sparsity_penalty', type=float, help='sparsity penalty weight', default=1e-3)

# ViT arguments
parser.add_argument('--b_dim', type=int, help='dimension of ViT', default=768)      
parser.add_argument('--out_dim', type=int, help='dimension of output', default=102)                    
parser.add_argument('--name', type=str, help='model name', default='ViT-SAE')              
parser.add_argument('--layer', type=int, help='layer of ViT to extract activations', default=-2)

# Training arguments
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-5)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--pretrain_epoch', type=str, help='loading pretrain epoch', default='5')
parser.add_argument('--use_finetuned_vit', action='store_true', default=False,
                    help='use fine-tuned ViT model instead of pretrained ViT')

# Optimization arguments
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--train_batch_size', type=int, help='training batch size', default=32)
parser.add_argument('--eval_batch_size', type=int, help='eval batch size', default=64)                    
parser.add_argument('--metric', type=str, help='eval metric', default='eval_accuracy')
# SAE specific arguments
parser.add_argument('--sae_lr', type=float, help='SAE learning rate', default=1e-3)
parser.add_argument('--sae_weight_decay', type=float, help='SAE weight decay', default=1e-5)
parser.add_argument('--sae_epochs', type=int, help='SAE training epochs', default=1000)
parser.add_argument('--sae_batch_size', type=int, help='SAE batch size', default=256)
parser.add_argument('--sae_patience', type=int, help='SAE early stopping patience', default=50)

# Evaluation arguments
parser.add_argument('--eval_only', action='store_true', default=False, 
                    help='only evaluate, do not train')
parser.add_argument('--save_activations', action='store_true', default=False,
                    help='save SAE activations for analysis')
parser.add_argument('--load_activations', action='store_true', default=False,
                    help='load pre-computed SAE activations')

# Logging arguments
parser.add_argument('--log_interval', type=int, help='logging interval', default=10)
parser.add_argument('--save_interval', type=int, help='model saving interval', default=100)
parser.add_argument('--eval_interval', type=int, help='evaluation interval', default=50)

# Device arguments
parser.add_argument('--device', type=str, help='device to use', default='cuda')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)

# Data augmentation arguments
parser.add_argument('--use_augmentation', action='store_true', default=False,
                    help='use data augmentation for stability evaluation')
parser.add_argument('--augmentation_strength', type=float, default=0.5,
                    help='strength of data augmentation')

# Memory optimization arguments
parser.add_argument('--use_cls_token_only', action='store_true', default=True,
                    help='use only CLS token instead of all tokens to save memory')
parser.add_argument('--eval_batch_size_large', type=int, default=512,
                    help='batch size for evaluation to balance memory and speed')

# Hierarchical evaluation (not applicable for SAE, but kept for compatibility)
parser.add_argument('--hierarchical', action='store_true', default=False,
                    help='enable hierarchical evaluation (not applicable for SAE)')

# Pretraining arguments
parser.add_argument('--pretrain', action='store_true', default=False,
                    help='enable pretraining mode (ViT or SAE)')
parser.add_argument('--pretrain_type', type=str, default='vit', choices=['vit', 'sae'],
                    help='type of pretraining: vit or sae')
parser.add_argument('--load_pretrained_sae', type=str, default=None,
                    help='path to load pretrained SAE model')
parser.add_argument('--load_pretrained_vit', type=str, default=None,
                    help='path to load pretrained ViT model')
parser.add_argument('--pretrain_save_path', type=str, default='../ckpt/pretrained_models',
                    help='path to save pretrained models')
parser.add_argument('--resume_pretrain', type=str, default=None,
                    help='path to resume pretraining from checkpoint')


# Wandb arguments
parser.add_argument('--use_wandb', action='store_true', default=True,
                    help='use Weights & Biases for experiment tracking')
parser.add_argument('--wandb_project', type=str, default='sae-vit-interpretability',
                    help='Wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Wandb entity/team name')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Wandb run name (defaults to task + timestamp)')
parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                    help='Tags for wandb run')
parser.add_argument('--wandb_notes', type=str, default='',
                    help='Notes for wandb run')
