import time
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
import os


# model_dir = "/models/checkpoints/m2mcheckpoint"
# model_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "m2mcheckpoint")

def generate_text(prompt, model, tokenizer):
   # model = M2M100ForConditionalGeneration.from_pretrained(model_dir)
   # tokenizer = M2M100Tokenizer.from_pretrained(model_dir, src_lang="en", tgt_lang="ne")
   tokenizer.src_lang = "en"
   encoded_hi = tokenizer(prompt, return_tensors="pt")
   generated_tokens = model.generate(**encoded_hi, num_return_sequences=2,forced_bos_token_id=tokenizer.get_lang_id("ne"))
   generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#    generated_text_2 = tokenizer.decode(generated_tokens[1], skip_special_tokens=True)
   return generated_text