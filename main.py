import warnings

# pandas의 DeprecationWarning과 torch의 UserWarning을 무시합니다.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

import argparse  # 명령줄 인자를 파싱하기 위한 모듈
from model_loaders import ModelLoaderFactory  # 모델 로더 팩토리 클래스
import pandas as pd  # 데이터 처리를 위한 라이브러리
import ast  # 문자열을 파이썬 객체로 변환하기 위한 모듈
from sklearn.metrics import f1_score, accuracy_score  # 성능 평가를 위한 메트릭
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 예측 결과 중 임계값 이상인 레이블을 추출하는 함수
def get_multi_label_prediction(predictions, threshold=0.3, ignore_label='neutral'):
    # 'neutral' 레이블을 제외한 예측 결과 필터링
    filtered_predictions = {k: v for k, v in predictions.items() if k != ignore_label}
    # 임계값(threshold) 이상인 레이블만을 추출
    return [label for label, prob in filtered_predictions.items() if prob >= threshold]

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="모델 로더")
    
    # 명령줄 인자 정의: 모델 유형, 모델 이름, 모델 작업, 평가 여부
    parser.add_argument('--model_type', type=str, help='모델 유형 (tflite, onnx, huggingface)')
    parser.add_argument('--model_name', type=str, required=True, help='모델 이름')
    parser.add_argument('--model_task', type=str, required=True, help='모델 작업')
    parser.add_argument('--eval', type=bool, help='모델 평가')

    args = parser.parse_args()

    # 모델 로더 팩토리 인스턴스 생성 및 로더 호출
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
    
    # 평가 모드일 경우
    if args.eval:
        # kist 데이터셋을 만들 때 기본 레이블을 아래와 같이 가정하고 만들었다. 
        # 'love'는 없는 레이블이지만, csv 파일의 레이블 값을 설정할 때 넣고 만들었다. 
        label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'disgust'] 
        test = pd.read_csv("data/KIST_TEST.csv")
        test['labels'] = test['labels'].apply(ast.literal_eval)

        actual_labels = []
        predicted_labels = []
        # correct_count = 0

        # 테스트 데이터에 대해 예측 진행
        for index, row in test.iterrows():
            input_text = row['text']
            predictions = model_loader.predict(input_text)
            predicted_label = max(predictions, key=predictions.get)
            actual_label = label[row['labels'][0]]

            actual_labels.append(actual_label)
            predicted_labels.append(predicted_label)

            if predicted_label != actual_label:
                print(f"Text: {input_text}")
                print(f"Actual Label: {actual_label}")
                print(f"Predicted Label: {predicted_label}")
                print("------")

        # 성능 지표 출력
        print(f"F1 Score: {f1_score(actual_labels, predicted_labels, average='weighted'):.4f}")
        print(f"Accuracy: {accuracy_score(actual_labels, predicted_labels):.4f}")

        exit()

    else: 
        while True: 
            text = input("감정을 알고 싶은 상황을 최대한 구체적으로 작성해주세요 : ")
            if text == "exit": 
                break
            text += prompt
            prediction = model_loader.predict(text)
            top_7_predictions = dict(list(prediction.items())[:7])
            formatted_output = "\n".join([f"{label}: {prob:.4f}" for label, prob in top_7_predictions.items()])
            print(formatted_output)

if __name__ == '__main__':
    main()
