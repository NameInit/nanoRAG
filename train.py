import torch
import os
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
from src import Transformer, create_masks, DEVICE, MODEL_CONFIG

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
conf = MODEL_CONFIG

if tokenizer.bos_token is None:
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
if tokenizer.eos_token is None:
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

dataset = load_dataset("sberquad", split='train[:5000]')

def preprocess_correct(example):
    input_text = f"Вопрос: {example['question']}\nКонтекст: {example['context']}\nОтвет:"
    
    answer = example['answers']['text'][0] if example['answers']['text'] else "Нет ответа"
    target_text = f"{answer}{tokenizer.eos_token}"
    
    tokenized_inp = tokenizer(
        input_text, 
        truncation=True, 
        padding='max_length', 
        max_length=conf['max_pos']
    )
    
    tokenized_tar = tokenizer(
        target_text, 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )
    
    return {
        'input_ids': tokenized_inp['input_ids'],
        'labels': tokenized_tar['input_ids']
    }

print("Подготовка данных...")
tokenized_ds = dataset.map(preprocess_correct, remove_columns=dataset.column_names)
tokenized_ds.set_format(type='torch', columns=['input_ids', 'labels'])

train_size = int(0.9 * len(tokenized_ds))
val_size = len(tokenized_ds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(tokenized_ds, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=conf['batch_size'])

model = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=tokenizer.vocab_size, 
    target_vocab_size=tokenizer.vocab_size, 
    max_pos=conf['max_pos']
).to(DEVICE)

print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

best_val_loss = float('inf')
patience = 3
patience_counter = 0

print("\nНачало обучения...")
print("="*60)

for epoch in range(30):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        inp = batch['input_ids'].to(DEVICE)
        tar = batch['labels'].to(DEVICE)
        
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)
        
        optimizer.zero_grad()
        predictions = model(inp, tar_inp, enc_mask, combined_mask, dec_mask)
        
        loss = criterion(predictions.view(-1, tokenizer.vocab_size), tar_real.reshape(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch['input_ids'].to(DEVICE)
            tar = batch['labels'].to(DEVICE)
            
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            
            enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)
            predictions = model(inp, tar_inp, enc_mask, combined_mask, dec_mask)
            
            loss = criterion(predictions.view(-1, tokenizer.vocab_size), tar_real.reshape(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nEpoch {epoch+1}/30")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        save_path = os.path.join("model", f"best_transformer_{timestamp}_loss_{avg_val_loss:.3f}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Лучшая модель сохранена: {save_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break
    
    scheduler.step(avg_val_loss)
    print("="*60)

print("\nОбучение завершено!")