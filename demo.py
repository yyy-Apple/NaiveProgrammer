# %%
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# %%
# change the position encoding if src_max_length is larger than 1024
# Get original model
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
sd = model.state_dict()

shorter_pos_embeds = sd['model.encoder.embed_positions.weight']  # 1024 + 2 embeddings

new_config = model.config
new_config.max_position_embeddings = 2048  # 2048 -> 2050
new_model = BartForConditionalGeneration(new_config)

correctly_shaped_pos_weight = new_model.model.encoder.embed_positions.weight
print(correctly_shaped_pos_weight)

# %%
for i in range(1):
    correctly_shaped_pos_weight[i * shorter_pos_embeds.shape[0]:
                                (i + 1) * shorter_pos_embeds.shape[0]] = shorter_pos_embeds

correctly_shaped_pos_weight[1 * shorter_pos_embeds.shape[0]:] = shorter_pos_embeds[2:, :]
# %%
sd['model.decoder.embed_positions.weight'] = torch.tensor(correctly_shaped_pos_weight.data)

sd['model.encoder.embed_positions.weight'] = torch.tensor(correctly_shaped_pos_weight.data)

new_model.load_state_dict(sd, strict=True)

print(new_model.model.encoder.embed_positions.weight)

# %%
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')


# %% Conditional Generation Example
# Mask filling only works for bart-large
from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
TXT = "My friends are <mask> but they eat too many carbs."

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
logits = model(input_ids)[0]

masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)

tokenizer.decode(predictions).split()
# ['good', 'great', 'all', 'really', 'very']



# %% Summarization example
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# see ``examples/summarization/bart/run_eval.py`` for a longer example
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

# Generate Summary
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])