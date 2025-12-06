
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
    NEW_MODEL_NAME = os.getenv("NEW_MODEL_NAME", "llama-navy-jurisdiction-lora-v2")
    QG_MODEL = os.getenv("QG_MODEL", "valhalla/t5-base-e2e-qg")
    GOOGLE_DRIVE_PATH = os.getenv("GOOGLE_DRIVE_PATH")
    
    # Text Generation / RAG settings
    CHUNK_SIZE = 1000
    
    # LoRA settings
    LORA_R = 32
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Training settings
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1
    GRAD_ACCUMULATION = 16
    MAX_SEQ_LENGTH = 1024
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data_storage")

# Ensure directories exist
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)
