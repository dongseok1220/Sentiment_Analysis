# pip install optimum[onnxruntime] --quiet
# help(FeaturesManager.check_supported_model_or_raise)

from transformers.onnx import FeaturesManager

model_identifier = "mobilebert" # model name

supported_features = FeaturesManager.get_supported_features_for_model_type(model_identifier)
print(supported_features)

from pathlib import Path
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import MobileBertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")

# load model and tokenizer
feature = 'masked-lm'

# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path("trfs-model.onnx")
)

'''
from pathlib import Path
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.onnx import FeaturesManager
from transformers import MobileBertForMaskedLM, AutoModelForSequenceClassification

model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# load config
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
onnx_config = model_onnx_config(model.config)

# export
onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=18,
        output=Path("trfs-model.onnx")
)
'''
