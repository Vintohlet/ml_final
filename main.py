from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from nltk.tokenize import word_tokenize
import pymorphy3
import nltk
import uvicorn
import nltk

try:
    nltk.data.find('tokenizers/punkt')
    print("punkt tokenizer is available!")
except LookupError:
    print("punkt tokenizer not found. Downloading...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
    print("punkt_tab resource is available!")
except LookupError:
    print("punkt_tab resource not found. Downloading...")
    nltk.download('punkt_tab')

# --- Инициализация FastAPI ---
app = FastAPI(
    title="API Классификации Текста",
    description="Классификация текста на русском языке с использованием XGBoost"
)

# --- Загрузка токенизатора ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# --- Загрузка модели и ресурсов ---
model = None
label_encoder = None
punctuation_marks_api = []
stop_words_api = []
morph_api = None

model_filename = "xgboost_text_classifier.joblib"

try:
    model_data = joblib.load(model_filename)
    model = model_data['model_pipeline']
    label_encoder = model_data['label_encoder']
    punctuation_marks_api = model_data['punctuation_marks']
    stop_words_api = model_data['stop_words']
    morph_api = pymorphy3.MorphAnalyzer()
    print(f"✅ Модель '{model_filename}' загружена успешно.")
except FileNotFoundError:
    print(f"❌ Файл модели '{model_filename}' не найден.")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")

# --- Предобработка текста ---
def preprocess_for_api(text):
    if model is None or morph_api is None or stop_words_api is None or punctuation_marks_api is None:
        print("❌ Ошибка предобработки: Ресурсы не загружены.")
        return ""

    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        preprocessed_text = []
        for token in tokens:
            if token and token not in punctuation_marks_api and token not in stop_words_api:
                try:
                    lemma = morph_api.parse(token)[0].normal_form
                    preprocessed_text.append(lemma)
                except:
                    continue
        return ' '.join(preprocessed_text)
    return ""

# --- Pydantic модели ---
class TextInput(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    direction: str
    probabilities: dict = None

# --- Эндпоинты ---
@app.get("/")
def read_root():
    return {"message": "Добро пожаловать в API Классификации Текста. Используйте POST /classify для классификации."}

@app.post("/classify", response_model=ClassificationResult)
def classify_text(input_data: TextInput):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    processed_text = preprocess_for_api(input_data.text)
    if not processed_text:
        raise HTTPException(status_code=400, detail="Невозможно обработать текст.")

    try:
        pred_encoded = model.predict([processed_text])[0]

        if pred_encoded not in range(len(label_encoder.classes_)):
            raise ValueError("Некорректный индекс предсказания.")

        prediction = label_encoder.inverse_transform([pred_encoded])[0]

        probabilities = {}
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba([processed_text])[0]
                for i, cls in enumerate(label_encoder.classes_):
                    probabilities[cls] = float(probs[i])
            except:
                probabilities = None
        else:
            probabilities = None

        return ClassificationResult(direction=prediction, probabilities=probabilities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

@app.get("/model_info")
def get_model_info():
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")
    return {
        "classes": label_encoder.classes_.tolist(),
        "model_type": "XGBoost + TF-IDF",
        "preprocessing": "Токенизация, удаление пунктуации/стоп-слов, лемматизация (pymorphy3)"
    }

# --- Запуск FastAPI на локальном сервере ---
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=6666)
        print("🚀 FastAPI приложение запущено на localhost:6666")
    except Exception as e:
        print(f"❌ Ошибка запуска FastAPI: {e}")
