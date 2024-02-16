# Hugging Face의 변환 모델, ONNX, TFLite 모델 로더를 가져옵니다.
from .hf_transformers import *
from .onnx_loader import *
from .tflite_loader import *
import torch
import torch.nn.functional as F

# 여러 모델을 조합하여 앙상블 학습을 수행하는 클래스입니다.
class Ensemble:
    def __init__(self, models):
        self.models = models  # 모델 리스트
        self.alpha = [0.96, 0.02, 0.02]  # 각 모델의 가중치

    def predict(self, input_text):
        # 예측할 라벨
        target_labels = ['sadness', 'joy', 'anger', 'fear', 'surprise', 'disgust']
        # 모든 예측값을 저장할 딕셔너리 초기화
        total_predictions = {label: 0 for label in target_labels}

        # 모든 모델에 대해 예측을 수행
        for model, model_alpha in zip(self.models, self.alpha):
            prediction = model.predict(input_text)
            # 각 라벨에 대한 확률 추출
            relevant_probs = [prediction.get(label, 0) for label in target_labels]
            # 소프트맥스를 통해 확률 정규화
            softmax_probs = F.softmax(torch.tensor(relevant_probs), dim=0).numpy()

            # 각 라벨에 대해 예측값 업데이트
            for label, probability in zip(target_labels, softmax_probs):
                total_predictions[label] += probability * model_alpha

        return total_predictions

# 모델 로더를 관리하는 팩토리 클래스
class ModelLoaderFactory:
    def __init__(self):
        # 사용할 모델 로더를 정의하는 딕셔너리
        self.loaders = {
            ('fill-mask', 'mobilebert', 'huggingface'): MobileBertForFillMask,
            ('fill-mask', 'mobilebert', 'onnx'): MobileBertForFillMaskONNX,
            ('classification', 'roberta', 'onnx') : RobertaClassification,
            ('classification', 'distilroberta', 'onnx') : DistilrobertaClassification,
            ('classification', 'distilbert', 'onnx') : DistilbertClassification,
            ('classification', 'distilroberta', 'huggingface') : DistilrobertaClassification,
        }

    def get_loader(self, task, model_name, model_type):
        if model_name == 'ensemble':
            return self._get_ensemble_loader(task, model_type)
        else:
            loader = self.loaders.get((task, model_name, model_type))
            if loader:
                print(f"Load {model_name}/{model_type}/{task}...", end="\n\n")
                return loader()
            else:
                raise ValueError(f"Unsupported task/model combination: {task}/{model_name}/{model_type}")

    def _get_classification_models(self, specific_model_type="onnx"):
        # 분류(classification) 작업과 특정 model_type을 위한 모든 모델을 로드합니다.
        classification_models = []
        for (task, model_name, model_type), loader in self.loaders.items():
            if task == 'classification' and (specific_model_type is None or model_type == specific_model_type):
                classification_models.append(loader())
        return classification_models

    def _get_ensemble_loader(self, task, model_type="onnx"):
        # Ensemble 클래스를 위해 로드된 분류 모델들을 전달합니다.
        if task != 'classification':
            raise ValueError("Ensemble is only available for classification tasks.")
        
        print("Load {} type model...".format(model_type), end="\n\n")
        return Ensemble(self._get_classification_models(model_type))
