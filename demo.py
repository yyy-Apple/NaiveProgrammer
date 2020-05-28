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