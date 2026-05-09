import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from src import Transformer, create_masks, MODEL_CONFIG, DEVICE
import os
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
conf = MODEL_CONFIG

dataset = load_dataset("sberquad", split='train[:50000]')

def preprocess(ex):
    inp_txt = f"Контекст: {ex['context']} Вопрос: {ex['question']}"
    tar_txt = ex['answers']['text'][0] if ex['answers']['text'] else ""
    token_inp = tokenizer(inp_txt, truncation=True, padding='max_length', max_length=conf['max_pos'])
    token_tar = tokenizer(tar_txt, truncation=True, padding='max_length', max_length=64)
    return {'input_ids': token_inp['input_ids'], 'labels': token_tar['input_ids']}

tokenized_ds = dataset.map(preprocess, batched=False)
tokenized_ds.set_format(type='torch', columns=['input_ids', 'labels'])
loader = DataLoader(tokenized_ds, batch_size=conf['batch_size'], shuffle=True)

model = Transformer(
    num_layers=conf['num_layers'], d_model=conf['d_model'], 
    num_heads=conf['num_heads'], dff=conf['dff'],
    input_vocab_size=tokenizer.vocab_size, 
    target_vocab_size=tokenizer.vocab_size, 
    max_pos=conf['max_pos']
).to(DEVICE)

torch.cuda.empty_cache()

optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'], weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

last_model = "model/ft_transformer_20260508-1045_ep14_loss_0.78.pth"
if os.path.exists(last_model):
    model.load_state_dict(torch.load(last_model, map_location=DEVICE))
    print("Load weight and can be start fine-tuning...")

model.train()
for epoch in range(100):
    epoch_loss = 0
    for batch in loader:
        inp = batch['input_ids'].to(DEVICE)
        tar = batch['labels'].to(DEVICE)
        
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)
        
        optimizer.zero_grad()
        out = model(inp, tar_inp, enc_mask, combined_mask, dec_mask)
        
        loss = criterion(out.view(-1, tokenizer.vocab_size), tar_real.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    print(f"Epoch: {epoch+1} | Loss: {avg_loss:.4f}")

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    torch.save(model.state_dict(), f"model/ft_transformer_{ts}_ep{epoch}_loss_{avg_loss:.2f}.pth")
