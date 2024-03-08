import hf_olmo
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    VisionEncoderDecoderModel,
    GraphormerForGraphClassification
)
import os, json
import torch.nn as nn
import dataclasses
from typing import Optional
import logging
from src import util

path = "/users/drw/transformers/src/transformers/models/"
pytorch_path = "/users/drw/pytorch/torch/nn/modules/"


@dataclasses.dataclass
class PyTorchModule:
    name: str
    # out_path: str
    # do_write = True
    module: Optional[nn.Module]
    text: list[str] = dataclasses.field(default_factory=list)
    children: list['PyTorchModule'] = dataclasses.field(default_factory=list)

    # def __post_init__(self):
    #     if self.module is not None and self.do_write:
    #         self.write_module_to_file()
    #
    # def write_module_to_file(self):
    #     if not os.path.exists(self.out_path):
    #         os.mkdir(self.out_path)
    #
    #     file_text = os.path.join(self.out_path, f"{self.name}.py")
    #     with open(file_text, 'w') as f:
    #         for line in self.text:
    #             f.write(line)

        # for



def _prepare_pytorch_classes() -> dict:
    class_to_text = {}

    modeling_files = []
    for folder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        for file in os.listdir(os.path.join(path, folder)):
            if file.startswith(f"modeling_{folder}"):
                model_file = os.path.join(path, folder, file)
                print(model_file)
                modeling_files.append(model_file)

    for file in os.listdir(pytorch_path):
        if os.path.isdir(file):
            continue
        model_file = os.path.join(pytorch_path, file)
        print(model_file)
        modeling_files.append(model_file)

    for file in modeling_files:
        _convert_pytorch_path_to_text(file, class_to_text)

    return class_to_text


def _pytorch_module_to_files(
        text_module: PyTorchModule,
        file_path: str,
        flat_folder: bool = True,
        escaped_json: bool = False):

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    file_text = os.path.join(file_path, f"{text_module.name}.py")
    with open(file_text, 'w') as f:
        for line in text_module.text:
            if escaped_json:
                f.write(json.dumps(line))
            else:
                f.write(line)

    for child in text_module.children:
        if flat_folder:
            _pytorch_module_to_files(child, file_path, flat_folder, escaped_json)
        else:
            _pytorch_module_to_files(child, os.path.join(file_path, child.name), flat_folder, escaped_json)


def _convert_pytorch_path_to_text(full_model_path: str, class_text: dict):
    """ Converts a model to text files, recursively"""
    previous_line_started: bool = False
    model_text = None

    with open(full_model_path, 'r') as f:
        for line in f:

            if not line.startswith("class") and not previous_line_started:
                continue

            if not previous_line_started:
                model_name = line.split("(")[0].split(" ")[1]
                model_text = [line]
            else:
                previous_line_started = False
                model_text.append(line)

            while True:
                line = f.readline()
                if len(line) == 0:
                    break

                if line[0] == " " or line[0] == "\n" or line[0] == "\t":
                    model_text.append(line)
                else:
                    class_text[model_name] = model_text
                    if line.startswith("class"):
                        model_name = line.split("(")[0].split(" ")[1]
                        model_text = [line]
                        previous_line_started = True
                    break
    if model_text and model_name not in class_text:
        class_text[model_name] = model_text


def get_module_name(module: nn.Module):
    # noinspection PyProtectedMember
    return module._get_name()


def breakdown_module(class_name: str, module: nn.Module, class_text: dict) -> Optional[PyTorchModule]:
    if class_name not in class_text:
        print(f"No answer for : {class_name}")
        return None

    text_module = PyTorchModule(name=class_name, module=module, text=class_text[class_name])

    for name, child in module.named_children():
        if isinstance(child, nn.ModuleList):

            module_list = PyTorchModule(name=name, module=None, text=[""])

            for sub_module in child:
                mod_child: PyTorchModule = breakdown_module(get_module_name(sub_module), sub_module, class_text)

                # mod_child.name = f"{name}_{mod_child.name}"
                if mod_child is not None:
                    module_list.children.append(mod_child)

            text_module.children.append(module_list)
            continue

        mod_child: PyTorchModule = breakdown_module(get_module_name(child), child, class_text)

        if mod_child is not None:
            # mod_child.name = f"{name}_{mod_child.name}"
            text_module.children.append(mod_child)

    return text_module


def _write_module_to_file(module: nn.Module, out_path: str):
    if not os.path.exists(out_path):
        os.mkdir("." + out_path)

    for key, value in module.named_parameters():
        print(key)
        sub_folder_plus_weight = os.path.join(out_path, f"{key}").replace(".", "/")

        os.makedirs("." + "/".join(sub_folder_plus_weight.split("/")[:-1]), exist_ok=True)

        with open("." + sub_folder_plus_weight, 'w') as f:
            f.write(util.numpy_to_json(value.detach().numpy()))



if __name__ == "__main__":
    do_write = False

    # model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_id = "microsoft/phi-1.5"
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5-16k")
    # model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")
    # model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
    # model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")

    # model_id = "FacebookAI/roberta-large-mnli"
    # model_id = "microsoft/phi-1_5"
    # folder = "/public/phi-1_5"
    model_id = "clefourrier/graphormer-base-pcqm4mv2"
    folder = "/public/graphormer-base-pcqm4mv2"
    model = GraphormerForGraphClassification.from_pretrained(model_id)
    print(model.config)
    # # exit(1)
    #
    if do_write:
        _write_module_to_file(model, folder)
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = None
    # vocab = sorted(tokenizer.get_vocab(), key=tokenizer.get_vocab().get)

    if do_write:
        with open(f".{folder}/vocabulary", 'w') as f:
            f.write(json.dumps(vocab))

    if do_write:
        exit(1)


    #
    class_text: dict = _prepare_pytorch_classes()

    module_name = get_module_name(model)
    print(module_name)
    module_text = breakdown_module(module_name, model, class_text)
    _pytorch_module_to_files(module_text, "./graphformer/", flat_folder=False, escaped_json=False)
    exit(1)
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    generated_ids = model.generate(encodeds, max_new_tokens=20, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
