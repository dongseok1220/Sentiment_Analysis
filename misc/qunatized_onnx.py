import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# ONNX 모델 파일 경로
print("Check you model path !")

onnx_model_path = 'trfs-model.onnx'
quantized_model_path = 'trfs-model-qunatized.onnx'

# ONNX 모델 양자화
quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8)