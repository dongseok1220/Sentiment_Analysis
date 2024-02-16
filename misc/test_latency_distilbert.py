import pandas as pd
import time
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnxruntime as ort
from os import cpu_count
import torch

cnt = 10

print("Latency 측정 코드 입니다. 현재 {} 데이터를 측정하고 있습니다.".format(cnt))

# 모델 이름과 캐시 디렉토리 지정
model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"

# 정규 모델 로드
pytorch_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to load the ONNX model
def load_onnx_model(model_filepath):
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = cpu_count()
    return ort.InferenceSession(model_filepath, options)

onnx_model = load_onnx_model("../model_loaders/saved_model/onnx/distilbert-base-emotions/trfs-model-quantized.onnx")

# 테스트 데이터 로드
test_df = pd.read_csv('../data/test.csv')

def predict_sentiment_regular(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    with torch.no_grad():
        outputs = pytorch_model(input_ids, attention_mask=attention_mask)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    return torch.argmax(probabilities, dim=1).item()

# 양자화된 모델을 사용하여 감정 예측 함수
def predict_sentiment_quantized(text):
    tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)
    input_feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    logits = onnx_model.run(None, input_feed)[0]
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)
    return torch.argmax(probabilities, dim=1).item()

# 시간 측정 및 추론 실행
def measure_latency_and_predict(func, sentences):
    start_time = time.time()
    predictions = [func(sentence) for sentence in sentences]
    end_time = time.time()
    return predictions, end_time - start_time

# 정규 모델 추론 및 시간 측정
regular_predictions, regular_latency = measure_latency_and_predict(predict_sentiment_regular, test_df['text'][:cnt])

# 양자화된 모델 추론 및 시간 측정
quantized_predictions, quantized_latency = measure_latency_and_predict(predict_sentiment_quantized, test_df['text'][:cnt])

# 평균 latency 계산
average_regular_latency = regular_latency / len(test_df[:cnt])
average_quantized_latency = quantized_latency / len(test_df[:cnt])

print(f"Average Latency for Regular Model: {average_regular_latency} seconds")
print(f"Average Latency for Quantized Model: {average_quantized_latency} seconds")


