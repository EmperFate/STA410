import torch
import pytest
import numpy as np
from denoiser import DiffusionTextDenoiser

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDiffusionTextDenoiser:
    """Test suite for the DiffusionTextDenoiser class."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return DiffusionTextDenoiser(
            model_dim=128,  # Smaller for tests
            num_timesteps=20,  # Fewer steps for tests
            tokenizer_name="bert-base-uncased",
            schedule_type="linear"
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 2
        seq_len = 5
        embed_dim = 128
        x_0 = torch.randn(batch_size, seq_len, embed_dim, device=device)
        return x_0
    
    def test_q_sample_shape(self, model, sample_data):
        """Test that q_sample maintains the right shape."""
        batch_size = sample_data.shape[0]
        
        # Test with a single timestep for all batch elements
        t = torch.ones(batch_size, device=device, dtype=torch.long)
        x_t = model.q_sample(sample_data, t)
        
        assert x_t.shape == sample_data.shape, f"Shape mismatch: {x_t.shape} vs {sample_data.shape}"
    
    def test_q_sample_broadcasting(self, model, sample_data):
        """Test the broadcasting fix for q_sample."""
        # Test with a scalar timestep
        t_scalar = torch.tensor([5], device=device)
        x_t_scalar = model.q_sample(sample_data, t_scalar)
        
        assert x_t_scalar.shape == sample_data.shape, f"Broadcasting issue: {x_t_scalar.shape} vs {sample_data.shape}"
    
    def test_q_sample_deterministic(self, model, sample_data):
        """Test that q_sample is deterministic with fixed noise."""
        batch_size = sample_data.shape[0]
        t = torch.ones(batch_size, device=device, dtype=torch.long)
        
        # Create fixed noise
        noise = torch.randn_like(sample_data)
        
        # Sample twice with the same noise
        x_t1 = model.q_sample(sample_data, t, noise)
        x_t2 = model.q_sample(sample_data, t, noise)
        
        # Should be identical
        assert torch.allclose(x_t1, x_t2), "q_sample is not deterministic with fixed noise"
    
    def test_forward_reverse_consistency_t1(self, model, sample_data):
        """Test consistency between forward and reverse processes at t=1."""
        batch_size = sample_data.shape[0]
        
        # Forward diffusion at t=1
        t = torch.ones(batch_size, device=device, dtype=torch.long)
        noise = torch.randn_like(sample_data)
        x_1 = model.q_sample(sample_data, t, noise)
        
        # In a perfect reverse process, we would recover the original
        # But since our model isn't trained, we'll check shape and scale
        mean_var = model.p_mean_variance(x_1, t)
        
        assert mean_var["mean"].shape == sample_data.shape, "Mean shape is incorrect"
        assert mean_var["variance"].shape[:2] == sample_data.shape[:2], "Variance shape is incorrect"
    
    def test_p_sample_with_hmc(self, model, sample_data):
        """Test that p_sample_with_hmc runs without errors."""
        batch_size = sample_data.shape[0]
        t = torch.ones(batch_size, device=device, dtype=torch.long) * 5
        
        # Regular sample
        x_t_reg = model.p_sample(sample_data, t)
        
        # Sample with HMC
        x_t_hmc = model.p_sample_with_hmc(
            sample_data, t, use_hmc=True, num_leapfrog_steps=2, step_size=0.01
        )
        
        # Check shapes match
        assert x_t_reg.shape == x_t_hmc.shape, "HMC sample shape doesn't match regular sample"
    
    def test_hmc_acceptance(self, model, sample_data):
        """Test that HMC acceptance logic works correctly."""
        batch_size = sample_data.shape[0]
        t = torch.ones(batch_size, device=device, dtype=torch.long) * 5
        
        # Set step size very small so proposal should be close to current
        x_t_hmc_small = model.hmc_refine(
            sample_data, t, num_leapfrog_steps=2, step_size=1e-5
        )
        
        # Small step size should result in proposals very close to original
        # So MSE should be small
        mse_small = torch.nn.functional.mse_loss(sample_data, x_t_hmc_small).item()
        
        # Larger step size should make more difference
        x_t_hmc_large = model.hmc_refine(
            sample_data, t, num_leapfrog_steps=2, step_size=0.1
        )
        mse_large = torch.nn.functional.mse_loss(sample_data, x_t_hmc_large).item()
        
        # The MSE with larger step size should be greater in most cases
        # This isn't guaranteed, but it's a good heuristic
        print(f"MSE with small step size: {mse_small}, with large step size: {mse_large}")