
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from ..utils.config import Config
from ..data.processor import DataProcessor
from ..model.engine import ModelEngine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Royal Saudi Navy Legal Expert API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
engine = ModelEngine()

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.3
    max_new_tokens: int = 250
    top_p: float = 0.8
    repetition_penalty: float = 1.2

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "Royal Saudi Navy Legal Expert System Active"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(Config.DATA_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger processing (async in real app, sync here for simplicity)
    try:
        text = DataProcessor.extract_text_from_pdf(file_path)
        chunks = DataProcessor.create_chunks(text)
        dataset = DataProcessor.generate_qa_dataset(chunks)
        dataset_path = os.path.join(Config.DATA_DIR, "dataset")
        dataset.save_to_disk(dataset_path)
        return {"status": "success", "message": f"Processed {len(dataset)} Q&A pairs from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def start_training():
    dataset_path = os.path.join(Config.DATA_DIR, "dataset")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="No processed dataset found. Upload PDF first.")
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Add formatting
        dataset_dict['train'] = dataset_dict['train'].map(DataProcessor.format_instruction)
        dataset_dict['test'] = dataset_dict['test'].map(DataProcessor.format_instruction)
        
        path = engine.train(dataset_dict['train'], dataset_dict['test'])
        
        # Load for inference immediately after training
        engine.load_model_for_inference(path)
        
        return {"status": "success", "message": "Training completed and model loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not engine.model:
        # Try loading default if exists
        engine.load_model_for_inference() # Will automatically check drive/local
        
    if not engine.model:
         raise HTTPException(status_code=400, detail="Model not ready. Please train first.")
            
    response = engine.generate_response(
        request.message,
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty
    )
    return ChatResponse(response=response)
