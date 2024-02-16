# pip install --upgrade pip --quiet
# pip install mediapipe-model-maker --quiet
# pip install datasets --quiet

import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import text_classifier
import pandas as pd

from datasets import load_dataset
from mediapipe_model_maker import quantization


emotions = load_dataset("emotion")

# 각 데이터 분할을 별도의 CSV 파일로 저장
for split in emotions.keys():
    dataframe = emotions[split].to_pandas()
    dataframe.to_csv(f"{split}.csv", index=False)

csv_params = text_classifier.CSVParams(
    text_column='text',  # 텍스트 열 이름을 지정
    label_column='label',  # 라벨 열 이름을 지정
    delimiter=','
) 

train_dataset = text_classifier.Dataset.from_csv(
    filename='train.csv', csv_params=csv_params)
validation_dataset = text_classifier.Dataset.from_csv(
    filename='validation.csv', csv_params=csv_params)

supported_model = text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER

# 훈련 하이퍼파라미터 설정
hparams = text_classifier.BertHParams(
    epochs=2,
    batch_size=64,
    learning_rate=2e-4,
    shuffle = True,
    export_dir="bert_exported_models"
)

# 모델 옵션 설정 (BertModelOptions 예시)
model_options = text_classifier.BertModelOptions(
    seq_len=128,
    dropout_rate=0.1,
    do_fine_tuning=True
)

# TextClassifierOptions 설정
options = text_classifier.TextClassifierOptions(
    supported_model=supported_model,
    hparams=hparams,
    model_options=model_options
)

bert_model = text_classifier.TextClassifier.create(train_dataset, validation_dataset, options)

test_dataset = text_classifier.Dataset.from_csv(
    filename='test.csv', csv_params=csv_params)

metrics = bert_model.evaluate(test_dataset)
print(f'Test loss:{metrics[0]}, Test accuracy:{metrics[1]}')


quantization_config = quantization.QuantizationConfig.for_dynamic()
bert_model.export_model(quantization_config=quantization_config)
bert_model.export_labels(export_dir=options.hparams.export_dir)