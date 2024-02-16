import onnxruntime as ort  # ONNX 모델을 실행하기 위한 라이브러리
from os import cpu_count  # CPU 코어 개수를 얻기 위한 함수
import numpy as np  # 수치 계산을 위한 라이브러리
from transformers import AutoTokenizer  # 토크나이저를 자동으로 로드하기 위한 클래스
import torch  # PyTorch 라이브러리
from tokenizers import Tokenizer  # Hugging Face의 Tokenizer
from .utils import format_predictions  # 예측 결과를 포매팅하는 유틸리티 함수
from enum import Enum, auto  # 열거형을 정의하기 위한 클래스

# ONNX 모델 최적화 수준을 정의하는 열거형
class OptimizationLevel(Enum):
    DISABLE_ALL = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 최적화 적용 안함
    ENABLE_BASIC = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # 기본 최적화 적용
    ENABLE_EXTENDED = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # 확장 최적화 적용
    ENABLE_ALL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 모든 최적화 적용

class ONNXModelLoader:
    """ONNX 모델을 로드하기 위한 베이스 클래스입니다."""
    def load_onnx_model(self, model_filepath, optimization_level: OptimizationLevel = OptimizationLevel.ENABLE_EXTENDED):
        """ONNX 모델을 로드하는 메서드입니다."""
        try:
            options = ort.SessionOptions()
            # 멀티스레딩을 위한 옵션 설정
            options.inter_op_num_threads, options.intra_op_num_threads = cpu_count(), cpu_count()

            options.graph_optimization_level = optimization_level.value

            providers = ["CPUExecutionProvider"]  # CPU를 이용한 실행 제공자 설정
            return ort.InferenceSession(path_or_bytes=model_filepath, sess_options=options, providers=providers)
        except Exception as e:
            raise RuntimeError(f"모델 로딩 중 오류 발생: {e}")

# https://huggingface.co/SamLowe/roberta-base-go_emotions
class RobertaClassification(ONNXModelLoader): 
    def __init__(self):
        model_filepath = "model_loaders/saved_model/onnx/roberta_base_go_emotions/model_quantized.onnx"
        self.model = self.load_onnx_model(model_filepath)
        self.tokenizer = Tokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        self.tokenizer.enable_padding(**{**self.tokenizer.padding, "length": None})
        # 예측할 라벨 목록
        self.labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

    def predict(self, sentence):
        """문장을 입력받아 감정을 예측하는 메서드입니다."""
        if not sentence:
            raise ValueError("입력 문장이 비어 있습니다.")

        sentences = [sentence]
        tokens_obj = self.tokenizer.encode_batch(sentences)
        input_feed_dict = {
            "input_ids": [t.ids for t in tokens_obj],
            "attention_mask": [t.attention_mask for t in tokens_obj]
        }
        logits = self.model.run(output_names=[self.model.get_outputs()[0].name], input_feed=input_feed_dict)[0]
        softmax_outputs = self.softmax(logits)
        return format_predictions(softmax_outputs[0], self.labels)

    @staticmethod
    def softmax(x):
        """소프트맥스 함수를 계산하는 정적 메서드입니다."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

# https://huggingface.co/google/mobilebert-uncased
class MobileBertForFillMaskONNX(ONNXModelLoader):
    def __init__(self):
        model_filepath ="model_loaders/saved_model/onnx/mobilebert-fill-mask/trfs-model.onnx"
        self.model = self.load_onnx_model(model_filepath)
        self.tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    def predict(self, sentence):
        if not sentence:
            raise ValueError("입력 문장이 비어 있습니다.")

        sentences = [sentence]
        tokens_obj = self.tokenizer(sentences, padding=True, return_tensors="np")

        input_ids = tokens_obj["input_ids"].astype(np.int64)
        attention_mask = tokens_obj["attention_mask"].astype(np.int64)
        token_type_ids = tokens_obj["token_type_ids"].astype(np.int64)

        output_names = [self.model.get_outputs()[0].name]
        input_feed_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        logits = self.model.run(output_names=output_names, input_feed=input_feed_dict)[0]

        mask_token_index = np.where(input_ids == self.tokenizer.mask_token_id)
        first_mask = mask_token_index[1][0]
        first_mask_logits = logits[0, first_mask]

        first_mask_logits_tensor = torch.tensor(first_mask_logits)

        probabilities = torch.nn.functional.softmax(first_mask_logits_tensor, dim=-1)
        
        # 상위 3개 토큰 및 해당 확률 추출
        top_tokens = probabilities.topk(3)
        tokens = [self.tokenizer.decode([token_id]) for token_id in top_tokens.indices]
        probs = top_tokens.values.tolist()

        # format_predictions 사용
        return format_predictions(probs, tokens)

# https://huggingface.co/ahmettasdemir/distilbert-base-uncased-finetuned-emotion
class DistilbertClassification(ONNXModelLoader): 
    def __init__(self):
        model_filepath ="model_loaders/saved_model/onnx/distilbert-base-emotions/trfs-model-quantized.onnx"
        self.model = self.load_onnx_model(model_filepath)
        self.tokenizer = AutoTokenizer.from_pretrained("ahmettasdemir/distilbert-base-uncased-finetuned-emotion")
        # 라벨 리스트 정의
        self.labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    def predict(self, sentence):
        if not sentence:
            raise ValueError("입력 문장이 비어 있습니다.")

        sentences = [sentence]
        tokens_obj = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="np")
        input_feed_dict = {
            "input_ids": tokens_obj["input_ids"].astype(np.int64),
            "attention_mask":  tokens_obj["attention_mask"].astype(np.int64)
        }
        logits = self.model.run(output_names=[self.model.get_outputs()[0].name], input_feed=input_feed_dict)[0]
        logits_tensor = torch.tensor(logits)
        probabilities = torch.nn.functional.softmax(logits_tensor, dim=-1)
       
        return format_predictions(probabilities[0], self.labels)
    
# @misc{hartmann2022emotionenglish,
#   author={Hartmann, Jochen},
#   title={Emotion English DistilRoBERTa-base},
#   year={2022},
#   howpublished = {\url{https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/}},
# }
class DistilrobertaClassification(ONNXModelLoader):
    def __init__(self):
        model_filepath = "model_loaders/saved_model/onnx/distilroberta-base/trfs-model-qunatized.onnx"
        self.model = self.load_onnx_model(model_filepath)
        self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        
        self.labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness','surprise']        

    def predict(self, sentence):
        if not sentence:
            raise ValueError("입력 문장이 비어 있습니다.")

        # 입력 문장 토크나이징 및 ONNX 모델 입력 형식에 맞게 변환
        tokens = self.tokenizer(sentence, return_tensors="np", padding=True, truncation=True)
        input_ids = tokens['input_ids'].astype(np.int64)
        attention_mask = tokens['attention_mask'].astype(np.int64)

        # ONNX 모델 추론 실행
        logits = self.model.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })[0]

        # 결과 처리
        softmax_outputs = self.softmax(logits)
        return format_predictions(softmax_outputs[0], self.labels)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
