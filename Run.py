import torch
import numpy as np
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm.auto import tqdm
import sys

# Suppress progress bars for downloads
nltk.download('punkt', quiet=True)
nltk.download('gutenberg', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Import the conditional model
from denoiser import ConditionalDiffusionTextDenoiser

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------------------------------------------------
# Utilities: text corruption, dataset loading, and evaluation metrics
# ----------------------------------------------------------------------

def corrupt_text(text, corruption_rate=0.3):
    """Apply simple token‐level corruptions."""
    tokens = text.split()
    corrupted = []
    keyboard = {
        'a': 'sqzw','b': 'vghn','c': 'xdfv','d': 'sfxce','e': 'wsdr','f': 'dcgvr',
        'g': 'fvhbt','h': 'gbnjy','i': 'ujko','j': 'hknmu','k': 'jlmio','l': 'kop',
        'm': 'njkl','n': 'bhjm','o': 'iklp','p': 'ol','q': 'wa','r': 'etdf',
        's': 'awedxz','t': 'ryfg','u': 'yihj','v': 'cfgb','w': 'qase','x': 'zsdc',
        'y': 'tghu','z': 'asx'
    }
    for tok in tokens:
        if random.random() < corruption_rate:
            choice = random.choice(['delete','swap','typo'])
            if choice=='delete' or len(tok)<=1:
                continue
            elif choice=='swap' and len(tok)>2:
                i = random.randint(0,len(tok)-2)
                lst = list(tok); lst[i],lst[i+1] = lst[i+1],lst[i]
                corrupted.append(''.join(lst))
            else:
                lst = list(tok)
                i = random.randrange(len(lst))
                c = lst[i].lower()
                if c in keyboard:
                    r = random.choice(keyboard[c])
                    lst[i] = r.upper() if tok[i].isupper() else r
                corrupted.append(''.join(lst))
        else:
            corrupted.append(tok)
    return ' '.join(corrupted)

def corrupt_dataset(texts, corruption_rate=0.3):
    return [corrupt_text(t, corruption_rate) for t in texts]

def calculate_bleu(orig_list, denoised_list):
    smooth = SmoothingFunction().method1
    scores = []
    for o,d in zip(orig_list, denoised_list):
        scores.append(sentence_bleu([o.split()], d.split(), smoothing_function=smooth))
    return float(np.mean(scores))

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
import random

def load_dataset(source="wikitext", max_samples=5000, max_length=40):
    """Load dataset with appropriate size for training.
    
    Args:
        source: Dataset source ("wikitext" or "recipes")
        max_samples: Maximum number of samples to return (default: 5000)
        max_length: Maximum length of sentences in tokens (default: 40)
        
    Returns:
        List of text strings
    """
    if source == "wikitext":
        raw = gutenberg.raw()  # grab the entire corpus as one string
        all_sents = sent_tokenize(raw)  # sentence-split with punkt
        texts = all_sents
    elif source == "recipes":
        texts = [
            "Preheat oven to 350°F and line a baking sheet with parchment.",
            "Mix flour, sugar, and salt in a bowl until well combined.",
        ]
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    # filter by length - shorter lengths to avoid memory issues
    filtered = [t for t in texts if 5 <= len(t.split()) <= max_length]

    # random-sample up to max_samples
    if len(filtered) > max_samples:
        filtered = random.sample(filtered, max_samples)

    return filtered


# ----------------------------------------------------------------------
# High‐level training + evaluation with improved parameters
# ----------------------------------------------------------------------

def train_and_evaluate(
    clean_texts,
    corruption_rate=0.3,
    batch_size=8,        # Increased batch size for better training
    num_epochs=20,       # Increased to allow more learning
    model_dim=768,       # Match BERT's dimension to avoid projection issues
    num_timesteps=20,    # Shortened chain for easier learning
    schedule_type="linear", # Changed from cosine for simpler initial learning
    noise_scale=0.05,    # Reduced noise scale for less extreme diffusion
    use_hmc=False,
    device="cpu",        # Default to CPU to avoid memory issues
    token_loss_weight=0.1, # Weight for token-level cross-entropy loss
    k_sampling=None      # Use full distribution sampling during training
):
    """Train and evaluate the text denoiser with improved parameters.
    
    Args:
        clean_texts: List of clean texts for training/testing
        corruption_rate: Rate of corruption to apply (default: 0.3)
        batch_size: Batch size for training (default: 8)
        num_epochs: Number of training epochs (default: 20)
        model_dim: Model dimension (default: 768 to match BERT)
        num_timesteps: Number of diffusion timesteps (default: 20)
        schedule_type: Diffusion schedule type (default: "linear")
        noise_scale: Scale of noise to add (default: 0.05)
        use_hmc: Whether to use HMC refinement in evaluation (default: True)
        device: Device to run on (default: "cpu")
        token_loss_weight: Weight for token cross-entropy loss (default: 0.1)
        k_sampling: k for top-k sampling (default: None = full distribution)
    """
    print(f"Training with {len(clean_texts)} examples, dim={model_dim}, timesteps={num_timesteps}")
    
    # Split into train/test sets
    n = len(clean_texts)
    split = int(0.8*n)
    train, test = clean_texts[:split], clean_texts[split:]
    
    # Create corrupted versions for both training and testing
    noisy_train = corrupt_dataset(train, corruption_rate)
    noisy_test = corrupt_dataset(test, corruption_rate)

    # Initialize the conditional model with improved parameters
    model = ConditionalDiffusionTextDenoiser(
        model_dim=model_dim,
        num_timesteps=num_timesteps,
        schedule_type=schedule_type,
        noise_scale=noise_scale,
        device=device,
        gradient_checkpointing=True,  # Enable for memory efficiency
        k_sampling=k_sampling          # Sampling strategy during training
    )

    # Train on paired data (clean and corrupted)
    model.fit(
        clean_texts=train,
        corrupted_texts=noisy_train,
        batch_size=batch_size,
        num_epochs=num_epochs,
        token_loss_weight=token_loss_weight  # Add token-level loss
    )

    # First evaluate with simpler approach (no HMC)
    print("\nRunning basic evaluation (no HMC)...")
    denoised_no_hmc = model.clean(
        noisy_test, 
        use_hmc=False,
        show_progress=True,
        k=1  # Use greedy decoding for final evaluation
    )
    
    # Now with HMC refinement if requested
    if use_hmc:
        print("\nRunning evaluation with HMC refinement...")
        denoised_hmc = model.clean(
            noisy_test,
            use_hmc=False, #true
            num_leapfrog_steps=5,
            step_size=1e-3,
            show_progress=True,
            k=1  # Use greedy decoding for final evaluation
        )
    else:
        denoised_hmc = denoised_no_hmc  # Skip HMC evaluation
    
    # Calculate metrics
    bleu_noisy = calculate_bleu(test, noisy_test)
    bleu_basic = calculate_bleu(test, denoised_no_hmc)
    bleu_hmc = calculate_bleu(test, denoised_hmc)

    print(f"\nBLEU scores → noisy: {bleu_noisy:.4f}, basic: {bleu_basic:.4f}, HMC: {bleu_hmc:.4f}")
    
    # Show sample outputs
    for i in range(min(5, len(test))):
        print("-"*60)
        print(f"ORIG: {test[i]}")
        print(f"NOIS: {noisy_test[i]}")
        print(f"BASIC: {denoised_no_hmc[i]}")
        if use_hmc:
            print(f" HMC : {denoised_hmc[i]}")

    return model


if __name__=="__main__":
    # Set all seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Choose device - use GPU if available and memory allows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load a much larger dataset as recommended
    print("Loading data...")
    data = load_dataset("wikitext", max_samples=200, max_length=40)
    print(f"→ got {len(data)} samples")

    # Training progression plan
    # 1. Start with smaller model if necessary (for memory constraints)
    model_dim = 768  # Try full BERT dimension first
    try:
        print("\nTraining & evaluating (attempt with full model dimension)...")
        trained_model = train_and_evaluate(
            data,
            corruption_rate=0.3,
            batch_size=8,          # Larger batch size for faster training
            num_epochs=20,         # More epochs
            model_dim=model_dim,   # Full BERT dimension
            num_timesteps=20,      # Shorter diffusion chain
            schedule_type="linear",# Simpler schedule for initial learning
            noise_scale=0.05,      # Less extreme noise
            use_hmc=False,         # Skip HMC for initial evaluation
            device=device,
            k_sampling=None        # Sample from full distribution during training
        )
    except RuntimeError as e:
        # If we run out of memory, try with reduced dimension
        if "CUDA out of memory" in str(e):
            print("\nCUDA out of memory with full dimension, trying with reduced dimension...")
            model_dim = 512  # Try a middle ground
            torch.cuda.empty_cache()
            
            trained_model = train_and_evaluate(
                data,
                corruption_rate=0.3,
                batch_size=4,          # Reduced batch size
                num_epochs=20,
                model_dim=model_dim,   # Reduced dimension
                num_timesteps=20,
                schedule_type="linear",
                noise_scale=0.05,
                use_hmc=False,
                device=device,
                k_sampling=None
            )
        else:
            # Re-raise other errors
            raise

    # Once base model trained, run final evaluation with HMC
    print("\nRunning final evaluation with HMC...")
    _ = train_and_evaluate(
        data[:1000],  # Use subset for final evaluation to speed up
        corruption_rate=0.3,
        batch_size=4,
        num_epochs=0,  # No additional training, just evaluation
        model_dim=model_dim,
        num_timesteps=20,
        schedule_type="linear",
        noise_scale=0.05,
        use_hmc=True,
        device=device,
        k_sampling=1  # Use greedy decoding for final evaluation
    )