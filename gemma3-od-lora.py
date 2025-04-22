import os; os.environ["CUDA_VISIBLE_DEVICES"]="3"

from config import Configuration
from utils import visualize_bounding_boxes, train_collate_fn, test_collate_fn

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from functools import partial

from peft import LoraConfig, get_peft_model, PeftType

cfg = Configuration()

train_dataset = load_dataset(cfg.dataset_id, split="train")
val_dataset = load_dataset(cfg.dataset_id, split="validation")
test_dataset = load_dataset(cfg.dataset_id, split="test")

print(f"{len(train_dataset)=}")
print(f"{len(val_dataset)=}")
print(f"{len(test_dataset)=}")

dir_name = "outputs_lora"
os.makedirs(dir_name, exist_ok=True)

data_sample = train_dataset[-100] # choose any other index to play around
visualize_bounding_boxes(
    data_sample["image"],
    data_sample["label_for_paligemma"],
    data_sample["width"],
    data_sample["height"],
    f"{dir_name}/sample.png"
)

processor = AutoProcessor.from_pretrained(cfg.model_id)

# pg2_tok = AutoTokenizer.from_pretrained(cfg.pg2_id)
# g3_tok = AutoTokenizer.from_pretrained(cfg.model_id)

# location_tokens = []

# values = list(pg2_tok.added_tokens_decoder.values())
# for v in values:
#     if "<loc" in v.content:
#         location_tokens.append(v.content)

# g3_tok.add_tokens(location_tokens)
# processor.tokenizer = g3_tok


train_collate_fn = partial(train_collate_fn, processor=processor, dtype=cfg.dtype)
test_collate_fn = partial(test_collate_fn, processor=processor, dtype=cfg.dtype)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=train_collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=train_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=test_collate_fn)

model = Gemma3ForConditionalGeneration.from_pretrained(
    cfg.model_id,
    torch_dtype=cfg.dtype,
    # device_map=cfg.device,
    # attn_implementation="eager", # As Sergio points out
)
model.requires_grad_(False)

lora_config = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj",],
    peft_type=PeftType.LORA,
)
lora_model = get_peft_model(model=model, peft_config=lora_config).to(cfg.device)
lora_model.print_trainable_parameters()

# model.resize_token_embeddings(len(processor.tokenizer)) # to integrate the new tokenizer

sample, sample_images = next(iter(test_dataloader))
sample = sample.to(cfg.device)
for key, value in sample.items():
    print(key, value.dtype, value.device)

# model.eval()
# generation = model.generate(**sample, max_new_tokens=100)
# decoded = processor.batch_decode(generation, skip_special_tokens=True)

# for s in decoded:
#     print(s)


params_to_train = list(filter(lambda x: x.requires_grad, lora_model.parameters())) # Note!
optimizer = torch.optim.AdamW(params_to_train, lr=cfg.learning_rate)

# Is the model activations deleted?

for epoch in range(1):
    for idx, batch in enumerate(train_dataloader):
        outputs = lora_model(**batch.to(cfg.device))
        loss = outputs.loss
        if idx % 100 == 0:
            print(f"Epoch: {epoch} Iter: {idx} Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Push to Hub
merged_model = lora_model.merge_and_unload()
merged_model.push_to_hub("ariG23498/g3-od-lora")
processor.push_to_hub("ariG23498/g3-od-lora")

merged_model.eval()
generation = merged_model.generate(**sample, max_new_tokens=100)
decoded = processor.batch_decode(generation, skip_special_tokens=True)

file_count = 0
for output_text, sample_image in zip(decoded, sample_images):
    im = sample_image[0]
    width, height = im.size
    visualize_bounding_boxes(im, output_text, width, height, f"{dir_name}/output_{file_count}.png")
    file_count += 1