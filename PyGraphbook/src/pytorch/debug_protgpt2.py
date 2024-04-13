from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline
protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")

# tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
# model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2")
print(protgpt2)


# inputs = tokenizer("<|endoftext|>", return_tensors="pt")
# generated = model(**inputs, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
# decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
# print(decoded)

#MGEAMGLTQPAVSRAVARLEERVGIRIFNRTARAITLTDEGRRFYEAVAPLLAGIEMHGYR\nVNVEGVAQLLELYARDILAEGRLVQLLPEWAD

sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=1, eos_token_id=0)

for sequence in sequences:
    print(sequence['generated_text'])