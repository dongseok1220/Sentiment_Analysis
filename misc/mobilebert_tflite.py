from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, MobileBertForMaskedLM, AutoTokenizer
from mediapipe.tasks import python
from mediapipe.tasks.python import text

model_path = '../model_loaders/saved_model/tflite/mobilebert-finetuned-emotion/model_english.tflite'

base_options = python.BaseOptions(model_asset_path=model_path)

classifier_options = text.TextClassifierOptions(
    base_options=base_options,
    max_results=6, 
)

classifier = python.text.TextClassifier.create_from_options(classifier_options)

def print_result(classification_result):
    label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    for classification in classification_result.classifications:
        for category in classification.categories:
            emotion_name = label[int(category.category_name)]
            score = category.score
            print(f"{emotion_name.capitalize()} : {score : .4f}")


INPUT_TEXT = input("영어로 상황을 입력해주세요 : ")
classification_result = classifier.classify(INPUT_TEXT)

print_result(classification_result)