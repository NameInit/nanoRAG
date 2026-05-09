import torch
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder
from .transformer import Transformer
from .loader import PDFLibrary
from .utils import create_masks

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

MODEL_CONFIG = {
    "num_layers": 4,
    "d_model": 256,
    "dff": 512,
    "num_heads": 8,
    "max_pos": 512,
    "dropout_rate": 0.1,
    "lr": 1e-4,
    "batch_size": 16
}