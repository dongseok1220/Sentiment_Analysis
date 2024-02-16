# !pip3 install onnx-simplifier --quiet

import onnx
from onnxsim import simplify

# 모델 로드
input_onnx_model = "trfs-model.onnx"

# 모델 단순화
simplified_model, check = simplify(input_onnx_model)

assert check, "Simplified ONNX model could not be validated"

# 단순화된 모델 저장
onnx.save(simplified_model, "simplified_model.onnx")
