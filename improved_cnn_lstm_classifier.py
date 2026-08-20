import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging
import math
import torch.nn.utils.prune as prune
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for computational efficiency with long sequences.
    Implements a simplified form of sparse attention to reduce quadratic complexity.

    BUG BRAVO FIX v6 (2026-02-27):
    - CRITICAL: Initialize query_scale/key_scale with varied values to break symmetry
    - Use position-dependent biases with full temporal resolution to ensure timesteps have different attention
    - Temperature scaling alone cannot create variance if inputs are symmetric
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, sparsity_factor: float = 0.5,
                 temperature: float = 0.35):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.temperature = temperature
        self.last_attention_weights = None
        self.max_seq_len = 50  # Maximum sequence length for position bias

        # BUG BRAVO FIX v6: Initialize with varied values to break symmetry across heads/positions
        # Create position-dependent scaling that varies across heads and positions
        self.query_scale = nn.Parameter(torch.randn(1, num_heads, 1, 1) * 0.1 + 1.0)
        self.key_scale = nn.Parameter(torch.randn(1, num_heads, 1, 1) * 0.1 + 1.0)

        # BUG BRAVO FIX v6: Add position bias that varies across timesteps
        # Shape: (1, num_heads, 1, max_seq_len) allows different bias per query position
        # This ensures attention weights differ across temporal positions
        self.position_bias = nn.Parameter(torch.randn(1, num_heads, 1, self.max_seq_len) * 0.1)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project and split into heads
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores with scaled dot-product
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # BUG BRAVO FIX v6: Apply input-dependent scaling AND position bias for diversity
        # Slice position bias to match current sequence length: (1, num_heads, 1, seq_len)
        position_bias_sliced = self.position_bias[:, :, :, :seq_len]
        # This breaks symmetry and ensures different heads/positions attend differently
        attention_scores = attention_scores * self.query_scale * self.key_scale + position_bias_sliced

        # Apply temperature scaling and softmax
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Store attention weights for extraction
        self.last_attention_weights = attention_weights.detach()

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out(attended)
        return output

    def get_attention_weights(self) -> torch.Tensor:
        """BUG BRAVO FIX: Extract stored attention weights for visualization."""
        return self.last_attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for better feature interaction modeling.
    
    BUG BRAVO FIX v5 (2026-02-27):
    - Add position-dependent bias to break temporal symmetry
    - Initialize with varied values to ensure attention variance across timesteps
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, use_sparse: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_sparse = use_sparse

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.last_attention_weights = None
        self.temperature = nn.Parameter(torch.tensor(0.5))

        # BUG BRAVO FIX v6 (2026-02-27): Position-dependent bias with full temporal resolution
        # Previous fix used (num_heads, 1, 1) which broadcasted equally to all positions
        # Now using (num_heads, 1, seq_len) to allow different biases per query position
        # This ensures attention weights vary across timesteps
        self.max_seq_len = 50  # Maximum sequence length for position bias
        self.position_bias = nn.Parameter(torch.randn(num_heads, 1, self.max_seq_len) * 0.1)

        if use_sparse:
            self.sparse_attention = SparseAttention(hidden_size, num_heads, sparsity_factor=0.5, temperature=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sparse:
            return self.sparse_attention(x)

        batch_size, seq_len, _ = x.shape

        # Project and split into heads
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with temperature scaling and position bias
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # BUG BRAVO FIX v6: Add position-dependent bias that varies per timestep
        # Slice position bias to match current sequence length: (num_heads, 1, seq_len)
        position_bias_sliced = self.position_bias[:, :, :seq_len]
        # Add bias to attention scores - this breaks temporal symmetry
        attention_scores = attention_scores / self.temperature.clamp(min=0.1) + position_bias_sliced
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Store attention weights for later extraction
        self.last_attention_weights = attention_weights.detach()

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out(attended)
        return output

    def get_attention_weights(self) -> torch.Tensor:
        """BUG BRAVO FIX: Extract stored attention weights for visualization."""
        if self.use_sparse and hasattr(self, 'sparse_attention'):
            return self.sparse_attention.get_attention_weights()
        return self.last_attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding to add sequence position information to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention, feed-forward network, and layer normalization.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, ff_hidden_size: int = None, dropout: float = 0.1, use_sparse_attention: bool = False):
        super().__init__()

        if ff_hidden_size is None:
            ff_hidden_size = hidden_size * 4

        self.attention = MultiHeadAttention(hidden_size, num_heads, use_sparse=use_sparse_attention)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer for network topology analysis.
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: node features [batch_size, num_nodes, input_dim]
        # adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        
        # Clamp input for numerical stability
        x = torch.clamp(x, min=-1e4, max=1e4)
        adj = torch.clamp(adj, min=0, max=1e4)

        # CRITICAL FIX: Ensure tensors are contiguous before batch matrix multiplication
        x = x.contiguous()
        adj = adj.contiguous()

        support = torch.bmm(x, self.weight)  # [batch_size, num_nodes, output_dim]
        
        # Clamp support for stability
        support = torch.clamp(support, min=-1e4, max=1e4)
        
        output = torch.bmm(adj, support)  # [batch_size, num_nodes, output_dim]
        
        # Clamp output for stability
        output = torch.clamp(output, min=-1e4, max=1e4)

        if self.bias is not None:
            output = output + self.bias
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for network topology analysis with attention mechanism.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.a = nn.Linear(2 * output_dim, 1, bias=False)

        self.dropout_layer = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: node features [batch_size, num_nodes, input_dim]
        # adj: adjacency matrix [batch_size, num_nodes, num_nodes]
        
        # Clamp input for numerical stability
        h = torch.clamp(h, min=-1e4, max=1e4)

        Wh = self.linear(h)  # [batch_size, num_nodes, output_dim]
        
        # Clamp transformed features
        Wh = torch.clamp(Wh, min=-1e4, max=1e4)

        # Compute attention coefficients
        batch_size, N = Wh.size(0), Wh.size(1)

        # Create attention matrix
        a_input = self._prepare_attentional_mechanism_input(Wh)  # [batch_size, N, N, 2 * output_dim]
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # [batch_size, N, N, 1] -> [batch_size, N, N]
        
        # Clamp attention logits for stability
        e = torch.clamp(e, min=-1e2, max=1e2)

        # Mask values before applying softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout_layer(attention)

        # CRITICAL FIX: Ensure tensors are contiguous before batch matrix multiplication
        # to prevent CUDA illegal memory access errors
        attention = attention.contiguous()
        Wh = Wh.contiguous()

        # Apply attention to features
        h_prime = torch.bmm(attention, Wh)  # [batch_size, N, output_dim]
        
        # Clamp output for stability
        h_prime = torch.clamp(h_prime, min=-1e4, max=1e4)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size()[1]  # number of nodes
        
        # Below, two matrices are created that contain embeddings in their rows where each row is repeated N times
        # The repeated rows are the features of the source node
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # [batch_size, N*N, output_dim]
        # The repeated columns are the features of the target node
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # [batch_size, N*N, output_dim]
        
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == [batch_size, N*N, output_dim]
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)  # [batch_size, N*N, 2*output_dim]
        
        # Reshape to [batch_size, N, N, 2*output_dim]
        return all_combinations.view(Wh.size(0), N, N, 2 * self.output_dim)


class NetworkTopologyEncoder(nn.Module):
    """
    Encoder for network topology using graph neural networks.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 128, num_layers: int = 2):
        super(NetworkTopologyEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GraphAttentionLayer(current_dim, hidden_dim))
            else:
                self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim))
            current_dim = hidden_dim
            
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # node_features: [batch_size, num_nodes, input_dim]
        # adjacency_matrix: [batch_size, num_nodes, num_nodes]
        
        x = node_features
        for layer in self.layers:
            x = layer(x, adjacency_matrix)
            x = self.dropout(x)
            
        # Global pooling to get graph-level representation
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        x = self.output_projection(x)  # [batch_size, output_dim]
        
        return x


class TemporalTransformer(nn.Module):
    """
    Transformer encoder specifically designed for temporal sequence modeling.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1, use_sparse_attention: bool = False):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, hidden_size * 4, dropout, use_sparse_attention)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_size]
        x = self.positional_encoding(x)  # Add positional information
        x = self.dropout(x)

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        return x

class ResidualBlock1D(nn.Module):
    """
    Residual block for 1D convolutions to improve gradient flow.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class AdvancedCNNSpatialExtractor(nn.Module):
    """
    Advanced CNN component with residual connections and dilated convolutions.
    """
    def __init__(self, input_channels: int, hidden_channels: int = 64):
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm1d(hidden_channels)
        
        # Residual blocks with increasing dilation rates
        self.res_block1 = ResidualBlock1D(hidden_channels)
        self.dilated_conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, 
                                      padding=2, dilation=2)  # Dilated conv
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.res_block2 = ResidualBlock1D(hidden_channels)
        self.dilated_conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, 
                                      padding=4, dilation=4)  # More dilation
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        
        self.res_block3 = ResidualBlock1D(hidden_channels * 2)
        self.dilated_conv3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, 
                                      padding=8, dilation=8)  # Even more dilation
        self.bn3 = nn.BatchNorm1d(hidden_channels * 4)

        self.pool = nn.AdaptiveAvgPool1d(16)  # Adaptive pooling to fixed size
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Block 1
        x = self.res_block1(x)
        x = F.relu(self.bn1(self.dilated_conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Block 2
        x = self.res_block2(x)
        x = F.relu(self.bn2(self.dilated_conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Block 3
        x = self.res_block3(x)
        x = F.relu(self.bn3(self.dilated_conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        return x

class ImprovedCNNLSTMClassifier(nn.Module):
    """
    Improved hybrid CNN-LSTM architecture for network packet classification.
    Features:
    - Advanced CNN with residual connections and dilated convolutions
    - Bidirectional LSTM with layer normalization
    - Multi-head attention mechanism
    - Enhanced regularization techniques
    - Transformer-based temporal modeling
    - Graph neural networks for network topology analysis
    - Uncertainty quantification
    - Memory-efficient gradient checkpointing
    - Sparse attention mechanisms
    
    CRITICAL MEMORY FIX: Default parameters reduced for CUDA OOM prevention
    """
    def __init__(self,
                 input_dim: int,
                 sequence_length: int = 10,
                 cnn_hidden: int = 16,  # REDUCED from 64 for memory
                 lstm_hidden: int = 32,  # REDUCED from 128 for memory
                 lstm_layers: int = 1,  # REDUCED from 2 for memory
                 transformer_hidden: int = 64,  # REDUCED from 256 for memory
                 transformer_layers: int = 1,  # REDUCED from 2 for memory
                 num_attention_heads: int = 2,  # REDUCED from 8 for memory
                 gnn_hidden: int = 16,  # REDUCED from 64 for memory
                 gnn_output_dim: int = 32,  # REDUCED from 128 for memory
                 gnn_layers: int = 1,  # REDUCED from 2 for memory
                 output_dim: int = 1,
                 dropout_rate: float = 0.5,
                 uncertainty_quantification: bool = False,  # DISABLED for memory
                 use_gnn: bool = False):  # DISABLED for memory
        """
        Initialize improved CNN-LSTM classifier.

        Args:
            input_dim: Number of input features per packet
            sequence_length: Number of packets in sequence
            cnn_hidden: Base number of CNN channels
            lstm_hidden: LSTM hidden state size
            lstm_layers: Number of LSTM layers
            transformer_hidden: Hidden size for transformer layers
            transformer_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            gnn_hidden: Hidden dimension for graph neural network
            gnn_output_dim: Output dimension for graph neural network
            gnn_layers: Number of GNN layers
            output_dim: Output dimension (1 for binary classification)
            dropout_rate: Dropout rate for regularization
            uncertainty_quantification: Whether to enable uncertainty estimation
            use_gnn: Whether to use the GNN encoder for topology analysis
        """
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.uncertainty_quantification = uncertainty_quantification
        self.use_gnn = use_gnn

        # Memory efficiency settings - disabled gradient checkpointing for numerical stability
        self.use_gradient_checkpointing = False
        self.enable_sparse_attention = True

        # Advanced CNN for spatial feature extraction
        self.cnn = AdvancedCNNSpatialExtractor(input_channels=input_dim, hidden_channels=cnn_hidden)

        # Calculate CNN output dimensions
        cnn_output_channels = cnn_hidden * 4
        cnn_output_length = 16  # Due to adaptive pooling

        # LSTM for temporal context with layer normalization
        self.lstm = nn.LSTM(
            input_size=cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0,
            proj_size=0  # Disable projection for simpler architecture
        )

        # Layer normalization for LSTM outputs
        self.lstm_layer_norm = nn.LayerNorm(lstm_hidden * 2)  # *2 for bidirectional

        # Transformer-based temporal modeling for enhanced sequence understanding
        self.temporal_transformer = TemporalTransformer(
            input_size=lstm_hidden * 2,  # *2 for bidirectional
            hidden_size=transformer_hidden,
            num_layers=transformer_layers,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            use_sparse_attention=True  # Enable sparse attention for efficiency
        )

        # Graph neural network for network topology analysis
        self.network_topology_encoder = NetworkTopologyEncoder(
            input_dim=input_dim,
            hidden_dim=gnn_hidden,
            output_dim=gnn_output_dim,
            num_layers=gnn_layers
        )

        # Fusion layer to combine temporal and topological features
        self.fusion_layer = nn.Sequential(
            nn.Linear(transformer_hidden + gnn_output_dim, transformer_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_hidden * 2, transformer_hidden),
            nn.ReLU()
        )

        # Additional attention mechanism after transformer
        self.transformer_attention = MultiHeadAttention(transformer_hidden, num_attention_heads, use_sparse=True)

        # Classification head with batch normalization and dropout
        self.classifier_layers = nn.Sequential(
            nn.Linear(transformer_hidden, transformer_hidden // 2),
            nn.BatchNorm1d(transformer_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_hidden // 2, transformer_hidden // 4),
            nn.BatchNorm1d(transformer_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(transformer_hidden // 4, output_dim)
        )

        # Uncertainty quantification branch - BUG ALFA FIX v5 (2026-02-27)
        # CRITICAL FIX: Deep sequential network was saturating outputs near 1.0
        # Solution: Use shallow linear network that preserves input variance
        # The uncertainty head must be sensitive to small changes in input variance
        if uncertainty_quantification:
            # Store transformer_hidden for uncertainty head
            self._uncertainty_input_dim = transformer_hidden

            # BUG ALFA FIX v5: Create projection layer for concatenated features
            # We concatenate: fused_output + seq_variance + seq_std
            # This gives us 3x transformer_hidden dimensions
            self.uncertainty_projection = nn.Linear(transformer_hidden * 3, transformer_hidden)

            # BUG ALFA FIX v5: Ultra-shallow uncertainty head to prevent saturation
            # Key insight: Deep networks with ReLU + Sigmoid saturate at extremes
            # Solution: Single linear layer with mild nonlinearity
            # This preserves sample-to-sample variance from the input features
            self.uncertainty_head = nn.Sequential(
                nn.Linear(transformer_hidden, transformer_hidden // 4),
                nn.LeakyReLU(negative_slope=0.01),  # More sensitive than ReLU
                nn.Linear(transformer_hidden // 4, 1),
                nn.Sigmoid()  # Bounds uncertainty to [0, 1]
            )

            # Initialize uncertainty head with variance-preserving weights
            self._init_uncertainty_head_v5()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.LSTM)):
                if isinstance(m, nn.LSTM):
                    # Special initialization for LSTM
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            param.data.fill_(0)
                            # Set forget gate bias to 1 for LSTM
                            n = param.size(0)
                            param.data[(n // 4):(n // 2)].fill_(1)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _init_uncertainty_head(self):
        """BUG ALFA FIX: Initialize uncertainty head with diverse weights to promote variance."""
        if not hasattr(self, 'uncertainty_head'):
            return

        for m in self.uncertainty_head.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialization promotes diverse feature responses
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    # Small random biases to break symmetry
                    m.bias.data.uniform_(-0.01, 0.01)

    def _init_uncertainty_head_v3(self):
        """BUG ALFA FIX v3: Initialize simplified uncertainty head to prevent saturation."""
        if not hasattr(self, 'uncertainty_head'):
            return

        for m in self.uncertainty_head.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization with moderate gain for stable signal propagation
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    # Small positive biases to keep ReLU neurons active
                    m.bias.data.fill_(0.1)

    def _init_uncertainty_head_v5(self):
        """BUG ALFA FIX v5: Initialize ultra-shallow uncertainty head for variance sensitivity."""
        if not hasattr(self, 'uncertainty_head'):
            return

        for i, m in enumerate(self.uncertainty_head.modules()):
            if isinstance(m, nn.Linear):
                # Use smaller gain for first layer to prevent saturation
                if i == 0:  # First linear layer
                    nn.init.xavier_uniform_(m.weight, gain=0.2)  # Reduced gain
                    # Initialize bias with small variance to break symmetry
                    m.bias.data.uniform_(-0.05, 0.05)
                else:  # Output layer
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    # Small bias to center the output
                    m.bias.data.fill_(0.0)

    def create_student_model(self, compression_ratio: float = 0.5):
        """
        Create a compressed student model for knowledge distillation.
        
        Args:
            compression_ratio: Ratio to compress the model (0.1 to 0.9)
        
        Returns:
            Student model with reduced parameters
        """
        # Calculate compressed dimensions
        compressed_cnn_hidden = max(16, int(self.cnn.initial_conv.out_channels * compression_ratio))
        compressed_lstm_hidden = max(32, int(self.lstm_hidden * compression_ratio))
        compressed_transformer_hidden = max(64, int(self.temporal_transformer.input_projection.out_features * compression_ratio))
        compressed_gnn_hidden = max(16, int(self.network_topology_encoder.layers[0].linear.in_features * compression_ratio))
        
        student_model = ImprovedCNNLSTMClassifier(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            cnn_hidden=compressed_cnn_hidden,
            lstm_hidden=compressed_lstm_hidden,
            lstm_layers=max(1, int(self.lstm_layers * compression_ratio)),
            transformer_hidden=compressed_transformer_hidden,
            transformer_layers=max(1, int(self.temporal_transformer.transformer_blocks.__len__() * compression_ratio)),
            num_attention_heads=max(2, int(self.temporal_transformer.transformer_blocks[0].attention.num_heads * compression_ratio)),
            gnn_hidden=compressed_gnn_hidden,
            gnn_output_dim=max(32, int(self.network_topology_encoder.output_projection.out_features * compression_ratio)),
            gnn_layers=max(1, int(len(self.network_topology_encoder.layers) * compression_ratio)),
            output_dim=self.classifier_layers[-1].out_features,
            dropout_rate=self.classifier_layers[3].p,  # Keep original dropout
            uncertainty_quantification=self.uncertainty_quantification
        )
        
        return student_model

    def compute_knowledge_distillation_loss(self, teacher_outputs, student_outputs, targets, 
                                         temperature: float = 3.0, alpha: float = 0.7):
        """
        Compute knowledge distillation loss combining soft and hard targets.
        
        Args:
            teacher_outputs: Outputs from the teacher model
            student_outputs: Outputs from the student model
            targets: Ground truth labels
            temperature: Temperature for softening probability distributions
            alpha: Weight for hard target loss
        
        Returns:
            Combined distillation loss
        """
        # Soft target loss (distillation loss)
        teacher_probs = torch.softmax(teacher_outputs / temperature, dim=1)
        student_log_probs = torch.log_softmax(student_outputs / temperature, dim=1)
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Hard target loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Combined loss
        total_loss = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * distillation_loss
        
        return total_loss, distillation_loss, hard_loss

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the improved hybrid model.

        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            adjacency_matrix: Adjacency matrix for graph neural network [batch_size, sequence_length, sequence_length]

        Returns:
            output: Classification logits [batch_size, output_dim]
            metadata: Intermediate activations for XAI analysis
        """
        batch_size = x.size(0)

        # Input validation and clamping for numerical stability
        x = torch.clamp(x, min=-1e6, max=1e6)

        # Reshape for CNN: [batch_size, input_dim, sequence_length]
        x_cnn = x.permute(0, 2, 1)

        # CNN feature extraction
        cnn_features = self.cnn(x_cnn)  # [batch_size, cnn_output_channels, cnn_output_length]
        
        # Clamp CNN outputs for stability
        cnn_features = torch.clamp(cnn_features, min=-1e4, max=1e4)

        # Reshape for LSTM: [batch_size, cnn_output_length, cnn_output_channels]
        lstm_input = cnn_features.permute(0, 2, 1)

        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(lstm_input)

        # Apply layer normalization to LSTM outputs
        lstm_output = self.lstm_layer_norm(lstm_output)
        
        # Clamp LSTM outputs for stability
        lstm_output = torch.clamp(lstm_output, min=-1e4, max=1e4)

        # Pass through temporal transformer for enhanced sequence modeling
        transformer_output = self.temporal_transformer(lstm_output)
        
        # Clamp transformer outputs for stability
        transformer_output = torch.clamp(transformer_output, min=-1e4, max=1e4)

        # Apply attention mechanism after transformer
        attended_output = self.transformer_attention(transformer_output)
        
        # Clamp attended outputs for stability
        attended_output = torch.clamp(attended_output, min=-1e4, max=1e4)

        # If GNN is enabled and adjacency matrix is provided, use graph neural network for topology analysis
        if self.use_gnn and adjacency_matrix is not None:
            # Use the input x as node features for the graph network
            topology_features = self.network_topology_encoder(x, adjacency_matrix)

            # Clamp topology features for stability
            topology_features = torch.clamp(topology_features, min=-1e4, max=1e4)

            # Global average pooling for transformer output
            pooled_temporal = torch.mean(attended_output, dim=1)  # [batch_size, transformer_hidden]

            # Concatenate temporal and topological features
            combined_features = torch.cat([pooled_temporal, topology_features], dim=1)  # [batch_size, transformer_hidden + gnn_output_dim]

            # Fuse the features
            fused_output = self.fusion_layer(combined_features)  # [batch_size, transformer_hidden]
        else:
            # Global average pooling to get fixed-size representation (skip GNN)
            fused_output = torch.mean(attended_output, dim=1)  # [batch_size, transformer_hidden]

        # Clamp fused output for stability
        fused_output = torch.clamp(fused_output, min=-1e4, max=1e4)

        # Classification
        output = self.classifier_layers(fused_output)

        # Validate output shape for binary classification
        # Output should be [batch_size, 1]
        if output.dim() != 2 or output.size(1) != 1:
            raise RuntimeError(f"Model output has invalid shape: {output.shape}. Expected [batch_size, 1]")

        # Check for NaN/Inf in output
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise RuntimeError(f"Model output contains NaN or Inf values. Output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")

        # Compute uncertainty if enabled - BUG ALFA FIX v4 (2026-02-27)
        # CRITICAL: LayerNorm was normalizing away sample-specific variance!
        # Solution: Remove LayerNorm and use direct feature concatenation
        # The uncertainty head needs raw variance signals, not normalized features
        uncertainty = None
        if self.uncertainty_quantification:
            # BUG ALFA FIX v4: Simplified uncertainty calculation without LayerNorm
            # Calculate sequence statistics (across time dimension) to capture per-sample variance
            seq_variance = torch.var(attended_output, dim=1)  # [batch_size, transformer_hidden]
            seq_std = torch.std(attended_output, dim=1)  # [batch_size, transformer_hidden]

            # Concatenate: [pooled | variance | std] = 3x transformer_hidden
            # This preserves sample-specific information that global pooling destroys
            uncertainty_input = torch.cat([fused_output, seq_variance, seq_std], dim=1)

            # Project to transformer_hidden dimension - NO LayerNorm!
            # LayerNorm was causing the flatline by normalizing away variance
            if hasattr(self, 'uncertainty_projection'):
                uncertainty_input = self.uncertainty_projection(uncertainty_input)
                uncertainty_input = F.relu(uncertainty_input)

            # Pass through uncertainty head
            uncertainty = self.uncertainty_head(uncertainty_input)

        # Store intermediate activations for XAI
        metadata = {
            'cnn_features': cnn_features,
            'lstm_output': lstm_output,
            'transformer_output': transformer_output,
            'attended_output': attended_output,
            'pooled_output': fused_output,
            'topology_features': topology_features if (self.use_gnn and adjacency_matrix is not None) else None,
            'uncertainty': uncertainty
        }

        return output, metadata

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN component for gradient checkpointing."""
        x_cnn = x.permute(0, 2, 1)
        cnn_features = self.cnn(x_cnn)
        return cnn_features

    def _forward_lstm(self, lstm_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM component for gradient checkpointing."""
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        lstm_output = self.lstm_layer_norm(lstm_output)
        return lstm_output

    def _forward_transformer(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer component for gradient checkpointing."""
        transformer_output = self.temporal_transformer(lstm_output)
        attended_output = self.transformer_attention(transformer_output)
        return transformer_output, attended_output

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the improved hybrid model.

        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            adjacency_matrix: Adjacency matrix for graph neural network [batch_size, sequence_length, sequence_length]

        Returns:
            output: Classification logits [batch_size, output_dim]
            metadata: Intermediate activations for XAI analysis
        """
        batch_size = x.size(0)

        # Input validation and clamping for numerical stability
        x = torch.clamp(x, min=-1e6, max=1e6)

        # Use gradient checkpointing for memory efficiency if enabled
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            # CNN feature extraction with gradient checkpointing
            cnn_features = checkpoint(self._forward_cnn, x, use_reentrant=False)
            
            # Clamp CNN outputs for stability
            cnn_features = torch.clamp(cnn_features, min=-1e4, max=1e4)

            # Reshape for LSTM: [batch_size, cnn_output_length, cnn_output_channels]
            lstm_input = cnn_features.permute(0, 2, 1)

            # LSTM processing with gradient checkpointing
            lstm_output = checkpoint(self._forward_lstm, lstm_input, use_reentrant=False)
            
            # Clamp LSTM outputs for stability
            lstm_output = torch.clamp(lstm_output, min=-1e4, max=1e4)

            # Transformer processing with gradient checkpointing
            transformer_output, attended_output = checkpoint(self._forward_transformer, lstm_output, use_reentrant=False)
            
            # Clamp transformer outputs for stability
            transformer_output = torch.clamp(transformer_output, min=-1e4, max=1e4)
            attended_output = torch.clamp(attended_output, min=-1e4, max=1e4)
        else:
            # Reshape for CNN: [batch_size, input_dim, sequence_length]
            x_cnn = x.permute(0, 2, 1)

            # CNN feature extraction
            cnn_features = self.cnn(x_cnn)  # [batch_size, cnn_output_channels, cnn_output_length]
            
            # Clamp CNN outputs for stability
            cnn_features = torch.clamp(cnn_features, min=-1e4, max=1e4)

            # Reshape for LSTM: [batch_size, cnn_output_length, cnn_output_channels]
            lstm_input = cnn_features.permute(0, 2, 1)

            # LSTM processing
            lstm_output, (hidden, cell) = self.lstm(lstm_input)

            # Apply layer normalization to LSTM outputs
            lstm_output = self.lstm_layer_norm(lstm_output)
            
            # Clamp LSTM outputs for stability
            lstm_output = torch.clamp(lstm_output, min=-1e4, max=1e4)

            # Pass through temporal transformer for enhanced sequence modeling
            transformer_output = self.temporal_transformer(lstm_output)
            
            # Clamp transformer outputs for stability
            transformer_output = torch.clamp(transformer_output, min=-1e4, max=1e4)

            # Apply attention mechanism after transformer
            attended_output = self.transformer_attention(transformer_output)
            
            # Clamp attended outputs for stability
            attended_output = torch.clamp(attended_output, min=-1e4, max=1e4)

        # If GNN is enabled and adjacency matrix is provided, use graph neural network for topology analysis
        if self.use_gnn and adjacency_matrix is not None:
            # Use the input x as node features for the graph network
            topology_features = self.network_topology_encoder(x, adjacency_matrix)

            # Clamp topology features for stability
            topology_features = torch.clamp(topology_features, min=-1e4, max=1e4)

            # Global average pooling for transformer output
            pooled_temporal = torch.mean(attended_output, dim=1)  # [batch_size, transformer_hidden]

            # Concatenate temporal and topological features
            combined_features = torch.cat([pooled_temporal, topology_features], dim=1)  # [batch_size, transformer_hidden + gnn_output_dim]

            # Fuse the features
            fused_output = self.fusion_layer(combined_features)  # [batch_size, transformer_hidden]
        else:
            # Global average pooling to get fixed-size representation (skip GNN)
            pooled_temporal = torch.mean(attended_output, dim=1)  # [batch_size, transformer_hidden]
            fused_output = pooled_temporal  # Use just the temporal features

        # Clamp fused output for stability
        fused_output = torch.clamp(fused_output, min=-1e4, max=1e4)

        # Classification
        output = self.classifier_layers(fused_output)
        
        # Validate output shape for binary classification
        # Output should be [batch_size, 1]
        if output.dim() != 2 or output.size(1) != 1:
            raise RuntimeError(f"Model output has invalid shape: {output.shape}. Expected [batch_size, 1]")
        
        # Check for NaN/Inf in output
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise RuntimeError(f"Model output contains NaN or Inf values. Output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")

        # Compute uncertainty if enabled - BUG ALFA FIX (2026-02-27)
        # CRITICAL: Must preserve sample-specific variance, not just pass pooled features
        # The uncertainty head needs variance signals from the sequence dimension
        uncertainty = None
        if self.uncertainty_quantification:
            # Calculate sequence statistics (across time dimension) to capture per-sample variance
            # attended_output shape: [batch_size, seq_len, transformer_hidden]
            seq_variance = torch.var(attended_output, dim=1)  # [batch_size, transformer_hidden]
            seq_std = torch.std(attended_output, dim=1)  # [batch_size, transformer_hidden]

            # Concatenate: [pooled | variance | std] = 3x transformer_hidden
            # This preserves sample-specific information that global pooling destroys
            uncertainty_input = torch.cat([fused_output, seq_variance, seq_std], dim=1)

            # Project to transformer_hidden dimension
            if hasattr(self, 'uncertainty_projection'):
                uncertainty_input = self.uncertainty_projection(uncertainty_input)
                uncertainty_input = F.relu(uncertainty_input)

            # Pass through uncertainty head
            uncertainty = self.uncertainty_head(uncertainty_input)

        # Store intermediate activations for XAI
        metadata = {
            'cnn_features': cnn_features,
            'lstm_output': lstm_output,
            'transformer_output': transformer_output,
            'attended_output': attended_output,
            'pooled_output': fused_output,
            'topology_features': topology_features if (self.use_gnn and adjacency_matrix is not None) else None,
            'uncertainty': uncertainty
        }

        return output, metadata

    def predict(self, x: torch.Tensor, adjacency_matrix: torch.Tensor = None) -> torch.Tensor:
        """Generate predictions with sigmoid activation for binary classification."""
        logits, _ = self.forward(x, adjacency_matrix)
        return torch.sigmoid(logits)

    def get_model_complexity(self) -> Dict[str, int]:
        """Get model complexity metrics for performance analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_parameters': sum(p.numel() for name, p in self.named_parameters() if 'cnn' in name),
            'lstm_parameters': sum(p.numel() for name, p in self.named_parameters() if 'lstm' in name),
            'transformer_parameters': sum(p.numel() for name, p in self.named_parameters() if 'transformer' in name),
            'attention_parameters': sum(p.numel() for name, p in self.named_parameters() if 'attention' in name),
            'classifier_parameters': sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name),
            'uncertainty_parameters': sum(p.numel() for name, p in self.named_parameters() if 'uncertainty' in name)
        }

    def prune_model(self, pruning_ratio: float = 0.2, method: str = 'magnitude'):
        """
        Prune the model to reduce computational complexity.
        
        Args:
            pruning_ratio: Fraction of weights to prune (0.0 to 1.0)
            method: Pruning method ('magnitude', 'random', 'l1_unstructured')
        """
        if method == 'magnitude':
            # Apply magnitude-based pruning to linear and conv layers
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    try:
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                        logger.info(f"Pruned {pruning_ratio*100}% of weights in {name}")
                    except Exception as e:
                        logger.warning(f"Could not prune {name}: {e}")
        elif method == 'random':
            # Apply random pruning
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    try:
                        prune.random_unstructured(module, name='weight', amount=pruning_ratio)
                        logger.info(f"Randomly pruned {pruning_ratio*100}% of weights in {name}")
                    except Exception as e:
                        logger.warning(f"Could not prune {name}: {e}")
        elif method == 'l1_unstructured':
            # Apply L1 unstructured pruning
            for name, module in self.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    try:
                        prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                        logger.info(f"L1 unstructured pruned {pruning_ratio*100}% of weights in {name}")
                    except Exception as e:
                        logger.warning(f"Could not prune {name}: {e}")
        else:
            logger.warning(f"Unknown pruning method: {method}")

    def remove_pruning(self):
        """
        Remove pruning reparameterization and restore original weights.
        """
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                try:
                    prune.remove(module, 'weight')
                    logger.info(f"Removed pruning from {name}")
                except ValueError:
                    # Module wasn't pruned
                    continue

    def get_sparsity(self) -> Dict[str, float]:
        """
        Get sparsity information for pruned layers.
        
        Returns:
            Dictionary with sparsity information for each layer
        """
        sparsity_info = {}
        total_params = 0
        total_zero_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight_orig'):
                    # This layer is pruned
                    mask = getattr(module, 'weight_mask')
                    zero_count = int((mask == 0).sum().item())
                    total_count = mask.numel()
                    sparsity = zero_count / total_count if total_count > 0 else 0
                    
                    sparsity_info[name] = sparsity
                    total_params += total_count
                    total_zero_params += zero_count
        
        if total_params > 0:
            sparsity_info['overall_sparsity'] = total_zero_params / total_params
        else:
            sparsity_info['overall_sparsity'] = 0.0
            
        return sparsity_info