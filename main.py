from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import joblib
from nltk.tokenize import word_tokenize
import nltk
import uvicorn
import re
from typing import Dict

# --- Инициализация NLTK ресурсов ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Инициализация FastAPI ---
app = FastAPI(
    title="AG News Classification API",
    description="API for classifying English news articles into 4 categories: World, Sports, Business, Sci/Tech"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Загрузка модели и ресурсов ---
model_data = None
model_filename = "agnews_xgboost_classifier.joblib"

try:
    model_data = joblib.load(model_filename)
    print(f"✅ Model '{model_filename}' loaded successfully.")
    print(f"Model classes: {model_data['label_encoder'].classes_}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# --- Предобработка текста (адаптированная для английского) ---
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    try:
        # Lowercase
        text = text.lower()
        
        # Tokenize and remove punctuation/stopwords
        tokens = word_tokenize(text)
        preprocessed = [
            token for token in tokens 
            if token not in model_data['punctuation_marks'] 
            and token not in model_data['stop_words']
            and len(token) > 1  # Remove single characters
        ]
        
        return ' '.join(preprocessed)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# --- Pydantic модели ---
class TextInput(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]

# --- Эндпоинты ---
@app.get("/")
def read_root():
    return {
        "message": "Welcome to AG News Classification API. Use POST /classify to classify news articles.",
        "categories": ["World", "Sports", "Business", "Sci/Tech"]
    }

@app.post("/classify", response_model=ClassificationResult)
async def classify_text(input_data: TextInput):
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess text
    processed_text = preprocess_text(input_data.text)
    if not processed_text:
        raise HTTPException(status_code=400, detail="Could not process the text")
    
    try:
        # Vectorize
        vector = model_data['vectorizer'].transform([processed_text])
        tfidf_vector = model_data['tfidf_transformer'].transform(vector)
        
        # Predict
        pred_proba = model_data['model'].predict_proba(tfidf_vector)[0]
        pred_class_idx = model_data['model'].predict(tfidf_vector)[0]
        pred_class = model_data['label_encoder'].classes_[pred_class_idx]
        
        # Prepare probabilities
        probabilities = {
            cls: float(pred_proba[i]) 
            for i, cls in enumerate(model_data['label_encoder'].classes_)
        }
        
        return ClassificationResult(
            category=pred_class,
            confidence=float(pred_proba[pred_class_idx]),
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
def get_model_info():
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost with TF-IDF",
        "classes": model_data['label_encoder'].classes_.tolist(),
        "accuracy": model_data.get('model_accuracy_on_test', 'unknown'),
        "preprocessing": "Lowercase, tokenization, punctuation/stopwords removal"
    }

# --- Запуск сервера ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)