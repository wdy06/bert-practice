import numpy as np
import torch

from transformers import BertJapaneseTokenizer, BertForMaskedLM

model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert_mlm = BertForMaskedLM.from_pretrained(model_name)

text = "今日は[MASK]へ行く。"
tokens = tokenizer.tokenize(text)
print(tokenizer(text))
print(tokens)

input_ids = tokenizer.encode(text, return_tensors="pt")
print(input_ids)
with torch.no_grad():
    output = bert_mlm(input_ids=input_ids)
    scores = output.logits

# ID列で'[MASK]'（IDは4）の位置を調べる
mask_position = input_ids[0].tolist().index(4)

print(scores.shape)

id_best = scores[0, mask_position].argmax(-1).item()
token_best = tokenizer.convert_ids_to_tokens(id_best)
token_best = token_best.replace("##", "")

text = text.replace("[MASK]", token_best)

print(text)
