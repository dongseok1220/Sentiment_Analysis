from transformers import AutoTokenizer, MobileBertForMaskedLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .utils import format_predictions

# https://huggingface.co/google/mobilebert-uncased
class MobileBertForFillMask:
    def __init__(self):
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.cache_dir ="model_loaders\saved_model\model_cache"

    def load_model(self):
        model = MobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        return tokenizer

    def predict(self, input_data):
        inputs = self.tokenizer(input_data, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        mask_logits = logits[0, mask_token_index]
        probabilities = torch.nn.functional.softmax(mask_logits[0], dim=0)

        # 토큰 ID와 예측 확률 매칭
        predicted_tokens_and_probs = []
        for token_id in mask_logits[0].topk(5).indices:
            token_label = self.tokenizer.decode([token_id])
            probability = probabilities[token_id].item()
            predicted_tokens_and_probs.append((token_label, probability))

        # 토큰과 확률을 분리하여 리스트로 저장
        labels, probs = zip(*predicted_tokens_and_probs)

        # format_predictions 함수를 사용하여 결과를 정렬하고 출력
        return format_predictions(probs, labels)

# @misc{hartmann2022emotionenglish,
#   author={Hartmann, Jochen},
#   title={Emotion English DistilRoBERTa-base},
#   year={2022},
#   howpublished = {\url{https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/}},
# }
class DistilrobertaClassification :
    def __init__(self):
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.cache_dir ="model_loaders\saved_model\model_cache"

    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        return model
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        return tokenizer

    def predict(self, input_data):
        inputs = self.tokenizer(input_data, truncation=True,padding=True)

        input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0)  # 배치 차원 추가
        attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)  # 배치 차원 추가

        self.model.eval()

        # 예측 실행
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # 첫 번째 예측 결과에 대한 확률과 레이블을 얻습니다.
        probabilities_list = probabilities[0].tolist()
        labels = self.model.config.id2label.values() 
        return format_predictions(probabilities_list, labels)