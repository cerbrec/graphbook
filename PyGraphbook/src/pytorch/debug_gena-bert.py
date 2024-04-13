from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('AIRI-Institute/gena-lm-bert-base', trust_remote_code=True)
gena_module_name = model.__class__.__module__
print(gena_module_name)
import importlib
# available class names:
# - BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
# - BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification,
# - BertForQuestionAnswering
# check https://huggingface.co/docs/transformers/model_doc/bert
cls = getattr(importlib.import_module(gena_module_name), 'BertForSequenceClassification')
print(cls)
model = cls.from_pretrained('AIRI-Institute/gena-lm-bert-base', num_labels=2)