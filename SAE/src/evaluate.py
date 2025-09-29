import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy


def to_numpy(x):
    """Convert tensor to numpy array"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def faithfulness(concept_train, pred_train, concept_test, pred_test, hard=False, prob_test=None):
    """
    Faithfulness metric: Train a linear classifier on SAE activations to predict labels
    Args:
        concept_train: SAE activations for training set (N_train, d_activation)
        pred_train: labels for training set (N_train,)
        concept_test: SAE activations for test set (N_test, d_activation)
        pred_test: labels for test set (N_test,)
        hard: whether to use hard labels
        prob_test: probability distribution for test set (for soft evaluation)
    Returns:
        score: classification accuracy
        soft_score: entropy-based score (if prob_test provided)
    """
    concept_train = to_numpy(concept_train)
    pred_train = to_numpy(pred_train) 
    concept_test = to_numpy(concept_test)
    pred_test = to_numpy(pred_test)
    
    # Use MLP classifier for faithfulness
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,), 
            activation='relu', 
            solver='adam', 
            alpha=0.0001, 
            batch_size=min(256, len(concept_train)),  # Adaptive batch size
            learning_rate='constant', 
            learning_rate_init=0.001, 
            power_t=0.5, 
            max_iter=200, 
            shuffle=True, 
            random_state=42,  # Fixed random state for reproducibility
            tol=0.0001, 
            verbose=False, 
            warm_start=False, 
            momentum=0.9, 
            nesterovs_momentum=True, 
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-08
        )
    )

    clf = pipe.fit(concept_train, pred_train)
    score = clf.score(concept_test, pred_test)

    return score, -1


def faithfulness_linear(concept_train, pred_train, concept_test, pred_test, hard=False, prob_test=None):
    """
    Faithfulness metric with linear classifier
    Args:
        concept_train: SAE activations for training set (N_train, d_activation)
        pred_train: labels for training set (N_train,)
        concept_test: SAE activations for test set (N_test, d_activation)
        pred_test: labels for test set (N_test,)
        hard: whether to use hard labels
        prob_test: probability distribution for test set (for soft evaluation)
    Returns:
        hard_score: classification accuracy
        soft_score: entropy-based score (if prob_test provided)
    """
    concept_train = to_numpy(concept_train)
    pred_train = to_numpy(pred_train)
    concept_test = to_numpy(concept_test)
    pred_test = to_numpy(pred_test)
    
    # Use logistic regression for linear faithfulness with optimized parameters
    pipe = make_pipeline(
        StandardScaler(), 
        LogisticRegression(
            random_state=42, 
            max_iter=1000,  # Reduced from 2500
            solver='liblinear',  # Faster for small datasets
            C=1.0
        )
    )
    clf = pipe.fit(concept_train, pred_train)
    hard_score = clf.score(concept_test, pred_test)
    
    if prob_test is not None:
        prob = clf.predict_proba(concept_test)
        soft_score = np.mean([entropy(prob[i], prob_test[i]) for i in range(len(prob))])
    else:
        soft_score = -1
    
    return hard_score, soft_score


def stability(concept_orig, concept_aug, compute=True):
    """
    Stability metric: Measure how much activations change under data augmentation
    Args:
        concept_orig: original SAE activations (N, d_activation)
        concept_aug: augmented SAE activations (N, d_activation)
        compute: whether to compute the metric
    Returns:
        stability_score: average relative change in activations
    """
    assert len(concept_orig.shape) == 2
    assert concept_orig.shape == concept_aug.shape
    
    concept_orig = to_numpy(concept_orig)
    concept_aug = to_numpy(concept_aug)
    
    # Compute relative change: ||theta - theta'|| / ||theta||
    delta = np.linalg.norm(concept_orig - concept_aug, axis=1) / np.linalg.norm(concept_orig, axis=1)
    
    return np.mean(delta)


def sparsity(concept):
    """
    Sparsity metric: Fraction of activations below threshold
    Args:
        concept: SAE activations (N, d_activation)
    Returns:
        sparsity_score: fraction of activations below threshold (0.1/d_activation)
    """
    assert len(concept.shape) == 2
    
    concept = to_numpy(concept)
    eps = 0.1 / concept.shape[1]  # threshold = 0.1 / d_activation
    
    return np.mean(concept < eps)


def parsimony(concept):
    """
    Parsimony metric: Number of activation dimensions
    Args:
        concept: SAE activations (N, d_activation)
    Returns:
        parsimony_score: number of activation dimensions
    """
    assert len(concept.shape) == 2
    return concept.shape[1]


def compute_all_metrics(concept_train, pred_train, concept_test, pred_test, 
                       concept_orig=None, concept_aug=None, prob_test=None):
    """
    Compute all SAE evaluation metrics with memory optimization
    Args:
        concept_train: SAE activations for training set
        pred_train: labels for training set
        concept_test: SAE activations for test set
        pred_test: labels for test set
        concept_orig: original activations for stability
        concept_aug: augmented activations for stability
        prob_test: probability distribution for test set
    Returns:
        metrics_dict: dictionary containing all metrics
    """
    import gc
    metrics = {}
    
    print(f"Computing metrics for data shapes: train={concept_train.shape}, test={concept_test.shape}")
    
    # Faithfulness metrics
    print("Computing faithfulness (MLP)...")
    metrics['faithfulness_mlp'], _ = faithfulness(concept_train, pred_train, concept_test, pred_test)
    gc.collect()  # Force garbage collection
    
    print("Computing faithfulness (Linear)...")
    metrics['faithfulness_linear'], metrics['faithfulness_soft'] = faithfulness_linear(
        concept_train, pred_train, concept_test, pred_test, prob_test=prob_test
    )
    gc.collect()  # Force garbage collection
    
    # Sparsity
    print("Computing sparsity...")
    metrics['sparsity'] = sparsity(concept_test)
    
    # Parsimony
    print("Computing parsimony...")
    metrics['parsimony'] = parsimony(concept_test)
    
    # Stability (if provided)
    if concept_orig is not None and concept_aug is not None:
        print(f"Computing stability for shapes: orig={concept_orig.shape}, aug={concept_aug.shape}")
        metrics['stability'] = stability(concept_orig, concept_aug)
        print(f"Stability: {metrics['stability']:.4f}")
    else:
        metrics['stability'] = -1
    
    # Final garbage collection
    gc.collect()
    
    return metrics


def print_metrics(metrics_dict):
    """Print all metrics in a formatted way"""
    print("\n" + "="*50)
    print("SAE EVALUATION METRICS")
    print("="*50)
    print(f"Faithfulness (MLP):     {metrics_dict['faithfulness_mlp']:.4f}")
    print(f"Faithfulness (Linear):  {metrics_dict['faithfulness_linear']:.4f}")
    if metrics_dict['faithfulness_soft'] != -1:
        print(f"Faithfulness (Soft):    {metrics_dict['faithfulness_soft']:.4f}")
    print(f"Sparsity:               {metrics_dict['sparsity']:.4f}")
    print(f"Parsimony:              {metrics_dict['parsimony']}")
    if metrics_dict['stability'] != -1:
        print(f"Stability:              {metrics_dict['stability']:.4f}")
    print("="*50)
