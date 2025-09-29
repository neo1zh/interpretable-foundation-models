import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import ViTImageProcessor
from datasets import load_dataset
import numpy as np
import os
from PIL import Image
from torch.utils.data import random_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from datasets import load_metric
from scipy.stats import entropy
import wandb
from datetime import datetime


def accuracy_score(labels, preds):
    """Compute accuracy score"""
    acc = (preds == labels).astype(np.float).mean()
    return acc


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels, preds = pred.predictions
    metric = load_metric('accuracy')
    return metric.compute(predictions=preds, references=labels)


class MyImageDataset(Dataset):
    """Dataset class for Image"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(MyImageDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        if self.transform:
            data = self.transform(data)
            img_to_tensor = transforms.ToTensor()
            # if data is not tensor
            if not isinstance(data, torch.Tensor):
                data = img_to_tensor(data)
        if self.normalize:
            data = self.normalize(data)
        
        return {'encodings': data, 'labels': self.labels[idx]}


def ensure_rgb_image(image):
    """Ensure image is in RGB format"""
    if hasattr(image, 'mode'):
        if image.mode != 'RGB':
            image = image.convert('RGB')
    return image


def load_dataset_by_task(task, data_path):
    """
    Load dataset based on task name
    Args:
        task: task name ('flower102', 'cub2011', 'cars', 'Color')
        data_path: path to dataset
    Returns:
        train_dataset, test_dataset, out_dim
    """
    img_size = (224, 224)
    
    if task == 'flower102':
        dataset_name = "nelorth/oxford-flowers"
        dataset = load_dataset(dataset_name)

        # 确保图像为RGB格式
        train_images = [ensure_rgb_image(img) for img in dataset['train']['image']]
        test_images = [ensure_rgb_image(img) for img in dataset['test']['image']]
        
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        train_inputs = processor(train_images, return_tensors="pt")
        test_inputs = processor(test_images, return_tensors="pt")
        train_dataset = MyImageDataset(train_inputs['pixel_values'], dataset['train']['label'])
        test_dataset = MyImageDataset(test_inputs['pixel_values'], dataset['test']['label'])
        out_dim = 102

    elif task == 'cub2011':
        dataset_name = "Donghyun99/CUB-200-2011"
        dataset = load_dataset(dataset_name)
        
        # 确保图像为RGB格式
        train_images = [ensure_rgb_image(img) for img in dataset['train']['image']]
        test_images = [ensure_rgb_image(img) for img in dataset['test']['image']]
        
        # 使用相同的processor确保一致性
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        train_inputs = processor(train_images, return_tensors="pt")
        test_inputs = processor(test_images, return_tensors="pt")
        train_dataset = MyImageDataset(train_inputs['pixel_values'], dataset['train']['label'])
        test_dataset = MyImageDataset(test_inputs['pixel_values'], dataset['test']['label'])
        out_dim = 200

    elif task == 'cars':
        dataset_name = "tanganke/stanford_cars"
        dataset = load_dataset(dataset_name)
        
        # 确保图像为RGB格式
        train_images = [ensure_rgb_image(img) for img in dataset['train']['image']]
        test_images = [ensure_rgb_image(img) for img in dataset['test']['image']]
        
        # 使用相同的processor确保一致性
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        train_inputs = processor(train_images, return_tensors="pt")
        test_inputs = processor(test_images, return_tensors="pt")
        train_dataset = MyImageDataset(train_inputs['pixel_values'], dataset['train']['label'])
        test_dataset = MyImageDataset(test_inputs['pixel_values'], dataset['test']['label'])
        out_dim = 196
        
    elif task == 'Color':
        # Define a transformation to convert the images to PyTorch tensors
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, ...]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1,1]
        ])

        # Load the images and labels
        dataset = []
        labels = []
        for class_dir in [os.path.join(data_path, 'Color/class0'), 
                         os.path.join(data_path, 'Color/class1')]:
            if os.path.exists(class_dir):
                for image_name in os.listdir(class_dir):
                    # Read image
                    image = Image.open(os.path.join(class_dir, image_name))
                    # Add to the lists
                    dataset.append(image)
                    labels.append(int(class_dir[-1]))  # class ID from the directory name

        # Convert lists to tensors
        labels = torch.tensor(labels)

        # Split into train and test sets
        # Pair up the data and labels
        paired_data = list(zip(dataset, labels))

        # Perform the split on the paired data
        train_size = int(0.8 * len(paired_data))  # 80% for training
        test_size = len(paired_data) - train_size
        train_data, test_data = random_split(paired_data, [train_size, test_size])

        train_images, train_labels = zip(*train_data)
        test_images, test_labels = zip(*test_data)

        # Convert the zipped data back to lists or tensors as needed
        train_images = list(train_images)
        train_labels = list(train_labels)
        test_images = list(test_images)
        test_labels = list(test_labels)

        # Create MyImageDataset instances
        train_dataset = MyImageDataset(train_images, train_labels, transform=transform)
        test_dataset = MyImageDataset(test_images, test_labels, transform=transform)
        out_dim = 2
        
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return train_dataset, test_dataset, out_dim


def image_augment(image, strength=0.5):
    """
    Apply data augmentation to images for stability evaluation
    Args:
        image: input image tensor (B, C, H, W)
        strength: augmentation strength
    Returns:
        augmented_image: augmented image tensor
    """
    b, c, h, w = image.shape
    contrast_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=strength),
        transforms.RandomResizedCrop(size=w, scale=(0.8, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.5 * strength,
                contrast=0.5 * strength,
                saturation=0.5 * strength,
                hue=0.1 * strength
            )
        ], p=0.8 * strength),
        transforms.RandomGrayscale(p=0.2 * strength),
        transforms.GaussianBlur(kernel_size=9),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_trans = contrast_transforms(image).cuda()
    return image_trans


def extract_embeddings(model, dataloader, device, layer, use_cls_token_only=True):
    """Extract embeddings from ViT model"""
    model.eval()
    embeddings = []
    labels = []
    preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            encodings = batch['encodings'].to(device)
            batch_labels = batch['labels'].to(device)
            
            # Get ViT outputs (只调用一次)
            logits, hidden, attention = model(encodings)
            
            # Extract predictions from logits
            batch_preds = logits.argmax(-1)  # (batch_size,)
            
            # Extract embeddings from specified layer
            all_states = model.ViT(encodings)['hidden_states']
            layer_embeddings = all_states[layer]  # (batch_size, seq_len, d_embedding)
            
            if use_cls_token_only:
                # 选择1：只使用CLS token (推荐) - 大幅减少内存使用
                cls_embeddings = layer_embeddings[:, 0, :]  # (batch_size, d_embedding)
                embeddings.append(cls_embeddings.cpu())
                labels.append(batch_labels.cpu())
                preds.append(batch_preds.cpu())
            else:
                # 选择2：使用所有tokens (需要复制labels) - 内存密集
                d_embedding = layer_embeddings.shape[2]
                seq_len = layer_embeddings.shape[1]
                all_embeddings = layer_embeddings.view(-1, d_embedding)
                embeddings.append(all_embeddings.cpu())
                # 复制labels和preds以匹配token数量
                expanded_labels = batch_labels.unsqueeze(1).repeat(1, seq_len).view(-1)
                expanded_preds = batch_preds.unsqueeze(1).repeat(1, seq_len).view(-1)
                labels.append(expanded_labels.cpu())
                preds.append(expanded_preds.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    return embeddings, labels, preds


def extract_augmented_activations(model, dataloader, device='cuda', layer=-2, augmentation_strength=0.5):
    """
    Extract SAE activations with data augmentation for stability evaluation
    Args:
        model: SAE model
        dataloader: data loader
        device: device to use
        layer: layer to extract activations from
        augmentation_strength: strength of data augmentation
    Returns:
        augmented_activations: augmented activations
        labels: corresponding labels
    """

    print("extract_augmented_activations")
    model.eval()
    augmented_activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            encodings = batch['encodings'].to(device)
            image_augmented = image_augment(encodings)
            logits_augmented, hidden_augmented, attention_augmented = model(image_augmented)
            augmented_activations.append(hidden_augmented.cpu())
    
    augmented_activations = torch.cat(augmented_activations, dim=0)
    # 只使用CLS token以节省内存
    augmented_activations = augmented_activations[:, 0, :]  # for cls token only
    return augmented_activations


def save_activations(activations, labels, save_path, epoch):
    """Save SAE activations to file"""
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, f'activations_epoch_{epoch}.npy'), activations)
    np.save(os.path.join(save_path, f'labels_epoch_{epoch}.npy'), labels)


def load_activations(load_path, epoch):
    """Load SAE activations from file"""
    activations = np.load(os.path.join(load_path, f'activations_epoch_{epoch}.npy'))
    labels = np.load(os.path.join(load_path, f'labels_epoch_{epoch}.npy'))
    return activations, labels


def create_data_loaders(train_dataset, test_dataset, train_batch_size=32, eval_batch_size=64, num_workers=4):
    """Create data loaders for training and evaluation"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_wandb(args):
    """
    Initialize Weights & Biases logging
    Args:
        args: command line arguments containing wandb configuration
    Returns:
        wandb run object
    """
    if not args.use_wandb:
        return None
    
    # Generate run name if not provided
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_run_name = f"{args.task}_{args.name}_{timestamp}"
    
    # Prepare wandb config
    wandb_config = {
        # Model parameters
        'd_embedding': args.d_embedding,
        'd_activation': args.d_activation if args.d_activation else args.d_embedding * args.expansion_factor,
        'expansion_factor': args.expansion_factor,
        'sparsity_penalty': args.sparsity_penalty,
        
        # Training parameters
        'sae_lr': args.sae_lr,
        'sae_weight_decay': args.sae_weight_decay,
        'sae_epochs': args.sae_epochs,
        'sae_batch_size': args.sae_batch_size,
        'sae_patience': args.sae_patience,
        
        # Data parameters
        'task': args.task,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'num_workers': args.num_workers,
        
        # ViT parameters
        'b_dim': args.b_dim,
        'layer': args.layer,
        'pretrain_epoch': args.pretrain_epoch,
        
        # System parameters
        'seed': args.seed,
        'device': args.device,
        
        # Evaluation parameters
        'eval_interval': args.eval_interval,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
    }
    
    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=wandb_config,
        tags=args.wandb_tags + [args.task, args.name],
        notes=args.wandb_notes,
        reinit=True
    )
    
    print(f"✅ Wandb initialized: {run.url}")
    return run


def log_wandb_metrics(metrics_dict, step=None, prefix=""):
    """
    Log metrics to wandb
    Args:
        metrics_dict: dictionary of metrics to log
        step: step number for logging
        prefix: prefix for metric names
    """
    if wandb.run is None:
        return
    
    # Add prefix to metric names
    if prefix:
        metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
    
    wandb.log(metrics_dict, step=step)


def log_wandb_model_artifacts(model, save_path, epoch=None, is_best=False):
    """
    Log model artifacts to wandb
    Args:
        model: PyTorch model to save
        save_path: path to save the model
        epoch: current epoch number
        is_best: whether this is the best model
    """
    if wandb.run is None:
        return
    
    # Create artifact
    artifact_name = f"sae_model_epoch_{epoch}" if epoch is not None else "sae_model"
    if is_best:
        artifact_name += "_best"
    
    artifact = wandb.Artifact(artifact_name, type="model")
    
    # Save model to file
    model_path = os.path.join(save_path, f"{artifact_name}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Add file to artifact
    artifact.add_file(model_path)
    
    # Log artifact
    wandb.log_artifact(artifact)
    
    print(f"✅ Logged model artifact: {artifact_name}")


def log_wandb_activations(activations, labels, save_path, epoch):
    """
    Log SAE activations to wandb as artifact
    Args:
        activations: SAE activations array
        labels: corresponding labels
        save_path: path to save activations
        epoch: current epoch
    """
    if wandb.run is None:
        return
    
    # Save activations
    activations_path = os.path.join(save_path, f'activations_epoch_{epoch}.npy')
    labels_path = os.path.join(save_path, f'labels_epoch_{epoch}.npy')
    
    np.save(activations_path, activations)
    np.save(labels_path, labels)
    
    # Create artifact
    artifact = wandb.Artifact(f"sae_activations_epoch_{epoch}", type="activations")
    artifact.add_file(activations_path)
    artifact.add_file(labels_path)
    
    # Log artifact
    wandb.log_artifact(artifact)
    
    print(f"✅ Logged activations artifact for epoch {epoch}")


def log_wandb_histogram(data, name, step=None):
    """
    Log histogram to wandb
    Args:
        data: data to create histogram from
        name: name of the histogram
        step: step number
    """
    if wandb.run is None:
        return
    
    wandb.log({name: wandb.Histogram(data)}, step=step)


def finish_wandb():
    """Finish wandb run"""
    if wandb.run is not None:
        wandb.finish()
        print("✅ Wandb run finished")
