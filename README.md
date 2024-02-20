# 자폐 아동을 위한 대화형 감정 분석 AI

## I. Intro 

### Objective 
자폐 스펙트럼 장애를 가진 아동들은 종종 감정을 이해하고 타인과의 사회적 상호작용에서 이를 적절히 표현하는 데 어려움을 겪는다. 자폐 아동의 감정 인식 및 표현 능력을 향상시키기 위해, 상호작용을 통해 감정을 분석하고 이해하는 데 도움을 주는 챗봇 AI '바로미'와 통합할 수 있는 라이브러리를 개발한다. 자폐 아동이 다양한 감정을 인식하고, 자신의 감정을 적절히 표현하는 데 필요한 지원을 제공하는 것을 목적으로 한다. AI는 자연어 처리 기술을 기반으로 한다.

### Overview
BERT 모델을 기반으로 하는 Huggingface의 다양한 모델들을 활용한다. **ONNX(Open Neural Network Exchange)** 형식으로 변환하여 모델 최적화 기법을 적용한다. 크게 3가지 task를 진행할 수 있다. 

첫 번째로, 입력된 문장을 학습된 데이터를 기반으로 적절한 감정을 확률 값과 함께 출력한다. 이때, Go-emotions 데이터 셋으로 학습된 모델의 경우 아래 RoBERTa 모델과 같이 12개의 긍정, 11개의 부정, 4개의 모호한 감정 표현과 중립 감정 표현으로 분석할 수 있다. Ensemble 모델의 경우 가장 기본적인 6가지 감정 표현 [화남, 슬픔, 기쁨, 공포, 혐오, 놀람]으로 예측한다. 

두 번째로, 입력된 문장 뒤에 올 감정을 토큰으로 예측한다. 입력된 문장 뒤에 프롬프트를 추가하여 [MASK] 토큰을 예측한다. 현재는 먼저 오는 [MASK] 토큰을 예측한다. 

마지막으로, 직접 구축한 KIST 데이터 셋을 테스트하여 특정 모델의 F1 score와 Accuracy를 확인할 수 있다. 

### 1. Classification
Input: “I got yelled to my boss today because I didn't do a good job. But it is not my fault!”

<table>
<tr>
<td>

**Output with RoBERTa:**  
- disappointment: 0.4206
- disapproval: 0.2511
- annoyance: 0.1870
- neutral: 0.0455
- sadness: 0.0243
- anger: 0.0171

</td>
<td>

**Output with Ensemble:**  
- sadness: 0.1689
- joy: 0.1646
- anger: 0.1734
- fear: 0.1641
- surprise: 0.1642
- disgust: 0.1647

</td>
</tr>
</table>


### 2. Fill-mask 
Input: “I got yelled to my boss today because I didn't do a good job. But it is not my fault!"  
Prompt: "I'm so [MASK] or felt [MASK]."

**Output with MobileBERT:**
- sorry: 0.1122
- angry: 0.0738
- tired: 0.0454
- hurt: 0.0409
- humiliated: 0.0263

## II. Architecture Details
Model은 `./model_loaders/saved_model`에 각각 `model_type`에 맞춰 저장한다.

모델은 각각 `onnx_loader.py`, `hf_transformers.py` 그리고 `tflite_loader.py`에서 `class`로 추가할 수 있다. 

### Example
```
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
```

`utils.py`에서는 모델들의 출력형식을 맞춰주는 `format_predictions` 메서드가 있다. 
```
def format_predictions(probabilities, labels):
    """ 확률과 라벨을 기반으로 정렬된 예측 결과를 반환합니다. """
    label_probabilities = {label: prob for label, prob in zip(labels, probabilities)}
    sorted_label_probabilities = dict(sorted(label_probabilities.items(), key=lambda item: item[1], reverse=True))

    return sorted_label_probabilities
```

`__init__.py`에서는 해당 파일들에서 추가한 모델들을 `class ModelLoaderFactory`를 통해 전체 관리한다. 

```
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
```
또한, `Ensemble`의 경우 각 모델의 가중치를 얼마나 줄 것인지 지정하는 `self.alpha` 파라미터가 존재한다. 현재 **onnx**모델은 3개이기 때문에 가중치 배열의 크기 역시 3이다. 
```
class Ensemble:
    def __init__(self, models):
        self.models = models  # 모델 리스트
        self.alpha = [0.96, 0.02, 0.02]  # 각 모델의 가중치

    def predict(self, input_text):
        # 예측할 라벨
        target_labels = ['sadness', 'joy', 'anger', 'fear', 'surprise', 'disgust']
        # 모든 예측값을 저장할 딕셔너리 초기화
        total_predictions = {label: 0 for label in target_labels}

    ... (cont'd)
```

`main.py`에서는 인자를 파싱하고, 모델을 로드하며 task를 수행한다. 
```
factory = ModelLoaderFactory()
    if (args.model_type == None) : 
        print("Default type is ONNX !")
        args.model_type = "onnx"
    model_loader = factory.get_loader(args.model_task, args.model_name, args.model_type)

    # 모델 유형에 따라 적절한 처리 진행
    if args.model_type == 'tflite':
        raise ValueError("Not implemented yet!")
    elif args.model_task == 'classification':
        prompt = ""
    elif args.model_task == 'fill-mask':
        prompt = "I'm so [MASK] or felt [MASK]."
    else:
        raise ValueError("지원되지 않는 모델 유형입니다.")
```

## III. DataSets & Performance
### KIST_TEST.csv
`label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'disgust']`
```
text,labels
I won the running race.,[ 1 ]
I feel good when I swing with my friend.,[ 1 ]
I am eating my favorite ice cream!,[ 1 ]
"I dropped my favorite ice cream, so I couldn't eat it anymore.",[ 0 ] 
The toy store has closed. ,[ 0 ]
My pet goldfish that I was raising has died.,[ 0 ]
My friend took away my toy!,[ 3 ]
My friend doodled on my drawing.,[ 3 ]
My friend is teasing me!,[ 3 ]
A burglar entered my house.,[ 4 ]
```
### Ensemble Model
- F1 Score: 0.7510
- Accuracy: 0.7531 


## IV. Environment Set-up
### Step1. Download this model
Use git clone or directly download. Then you can get a below model hierarchy.
```
├─main.py
│  
├─data
│      
├─misc
│
├─requirements.txt
│
├─model_loaders
│  │  hf_transformers.py
│  │  onnx_loader.py
│  │  tflite_loader.py
│  │  utils.py
│  │  __init__.py
│  │
│  ├─saved_model
│  │  ├─onnx
│  │  │  ├─distilbert-base-emotions
│  │  │  ├─mobilebert-fill-mask
│  │  │  └─roberta_base_go_emotions
│  │  │
│  │  └─tflite
│  │      └─mobilebert-finetuned-emotion
```
### Step2. Create anaconda virtual environment
```
(base) > conda create -n env_name
(base) > conda activate env_name
(env_name) > conda install python=3.10.13
(env_name) > pip install -r requirements.txt
```

### Step3. Download the model
You have to download the saved_model folder from this [link](https://drive.google.com/drive/folders/1RQvoMwzwbknMwGe--8Jlfnhbmq99EYKx?usp=sharing) and place it inside the model_loaders folder.   

## V. Quick Start
### Command

`model_type`은 생략할 수 있으며, 기본값은 **onnx**이다. 
1. `python main.py --model_type huggingface --model_name mobilebert --model_task fill-mask`
    ```
    Load mobilebert/huggingface/fill-mask...

    감정을 알고 싶은 상황을 최대한 구체적으로 작성해주세요 : I urgently went to the public restroom. A toilet was very dirty. Fortunately, the adjacent toilet was clean.
    embarrassed: 0.1198
    clean: 0.0644
    tired: 0.0630
    sorry: 0.0623
    ashamed: 0.0342
    ```
2. `python main.py --model_name ensemble --model_task classification`
    ```
    Default type is ONNX !
    Load onnx type model...

    감정을 알고 싶은 상황을 최대한 구체적으로 작성해주세요 : I urgently went to the public restroom. A toilet was very dirty. Fortunately, the adjacent toilet was clean.
    sadness: 0.1716
    joy: 0.1644
    anger: 0.1660
    fear: 0.1640
    surprise: 0.1631
    disgust: 0.1710
    ```
3. `python main.py --model_name ensemble --model_task classification --eval`  
    예측이 틀린 경우 원본 텍스트를 출력한다. 
    ```
    Default type is ONNX !
    Load onnx type model...

    Text: My friend doodled on my drawing.
    Actual Label: anger
    Predicted Label: disgust
    ------
    Text: My friend is teasing me!
    Actual Label: anger
    Predicted Label: joy
    ------
    ```

## VI. Miscellaneous
> Note: You may need to install external libraries to run the files below.

`convert2onnx.py`: 
- huggingface 모델을 onnx 형식으로 변환하는 파일  

`customize_mobilebert.py`:  
- Google MediaPipe에서 제공하는 mobilebert classifier를 자신만의 데이터셋으로 finetuning 하는 파일

`mobilebert_tflite.py`: 
- Google MediaPipe를 이용해 tensorflowlite 형식의 mobilebert를 실행할 수 있는 파일  

`quantized_onnx.py`:  
- onnx 에서 제공하는 모델을 경량화 하는 파일 

`simplifier_onnx.py`:
- onnx 에서 제공하는 모델을 심플하게 만드는 파일  

`test_latency_distilbert.py`:

- onnx 형식 모델과 기존 huggingface 모델의 latency를 측정하는 파일 
```  
Average Latency for Regular Model: 0.0478 seconds   
Average Latency for Quantized Model: 0.023 seconds
```


## VII. Requirements
```
python=3.10.13
transformers==4.37.2
torch==2.2.0
onnx==1.15.0
onnxruntime==1.17.0
onnxruntime-tools==1.7.0
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.3
pyarrow==15.0.0
mediapipe==0.10.9
```

## VIII. Reference
[ONNX](https://onnx.ai/)  
[huggingface](https://huggingface.co/)  
[mediapipe](https://developers.google.com/mediapipe)  
```
@misc{hartmann2022emotionenglish,
  author={Hartmann, Jochen},
  title={Emotion English DistilRoBERTa-base},
  year={2022},
  howpublished = {\url{https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/}},
}
```