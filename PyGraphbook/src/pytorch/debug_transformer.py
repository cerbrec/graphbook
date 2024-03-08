from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering
)
import os, json
import torch.nn as nn
import dataclasses
from typing import Optional
import logging
from src import util

def zephyr():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

    messages = [
        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    generated = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


def tinyllama():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.2"
    # model_id = "daryl149/llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Write vocab to file, should be a single list of strings
    # with open("vocabulary", 'w') as f:
    #     vocab = tokenizer.get_vocab()
    #     # Sort vocab by values into list of keys
    #     vocab = sorted(vocab, key=vocab.get)
    #     f.write(json.dumps(tokenizer.get_vocab()))

    model = AutoModelForCausalLM.from_pretrained(model_id)


    prompt = "How to get in a good university?"
    formatted_prompt = f"<|im_start|>user {prompt} <|im_end|> <|im_start|>assistant"
        # f"<|im_start|> user {prompt} <|im_end|> <|im_start|> assistant\n"
    # )

    print(formatted_prompt)

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    print(inputs)
    # Decode each token
    for token in inputs["input_ids"][0]:
        print(tokenizer.decode(token))
    # print(tokenizer.decode(inputs["input_ids"][0]))

    # exit(1)
    generated = model.generate(**inputs, do_sample=True, num_return_sequences=1, repetition_penalty=1.1, top_k=50, top_p=0.9, max_new_tokens=256)
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


def debug_roberta():
    from transformers import AutoTokenizer
    model_id = "FacebookAI/roberta-base"
    model = AutoModel.from_pretrained(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    generated = model(**inputs)

    print(generated)


def debug_phi_1_5():
    model_id = "microsoft/phi-1_5"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    x = '''def print_prime(n): """ Print all primes between 1 and n """'''
    inputs = tokenizer(x, return_tensors="pt", return_attention_mask=False)

    # Get back individual tokens
    print(inputs["input_ids"])
    print([tokenizer.decode(token) for token in inputs["input_ids"][0]])

    model = AutoModelForCausalLM.from_pretrained(model_id)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

if __name__ == "__main__":

    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # tinyllama()
    debug_roberta()
    # debug_phi_1_5()

    # zephyr()
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    #
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-chat")
    #
    # batch_encoding = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # # result = model(batch_encoding.data["input_ids"])
    # # print(model)
    #
    # generated_ids = model.generate(
    #     input_ids=batch_encoding.data["input_ids"],
    #     attention_mask=batch_encoding.data["attention_mask"],
    #     max_length=32,
    #     repetition_penalty=2.5,
    #     length_penalty=1.0,
    # )
    #
    # result = tokenizer.decode(
    #     generated_ids[0],
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=True)
    #
    # # output = tokenizer.decode(result[0])
    # print(result)
