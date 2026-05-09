import torch
from transformers import AutoTokenizer
from src import Transformer, create_masks, PDFLibrary, DEVICE, MODEL_CONFIG

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
conf = MODEL_CONFIG

model = Transformer(
    num_layers=conf['num_layers'], d_model=conf['d_model'], 
    num_heads=conf['num_heads'], dff=conf['dff'],
    input_vocab_size=tokenizer.vocab_size, 
    target_vocab_size=tokenizer.vocab_size, 
    max_pos=conf['max_pos']
).to(DEVICE)

model.load_state_dict(torch.load("model/ft_transformer_20260508-1045_ep14_loss_0.78.pth", weights_only=True))
model.eval()

def answer_question(question, pdf_path):
    lib = PDFLibrary(pdf_path)
    context = lib.find_context(question, n=1)[0]
    
    input_text = f"Контекст: {context} Вопрос: {question}"
    inp = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    
    output = torch.tensor([[tokenizer.cls_token_id]]).to(DEVICE)
    
    for i in range(50): # max 50 words in ans
        enc_mask, combined_mask, dec_mask = create_masks(inp, output)
        
        with torch.no_grad():
            predictions = model(inp, output, enc_mask, combined_mask, dec_mask)
        
        prediction = predictions[:, -1:, :]
        predicted_id = torch.argmax(prediction, axis=-1)
        
        output = torch.cat([output, predicted_id], dim=-1)
        
        if predicted_id == tokenizer.sep_token_id:
            break

    return tokenizer.decode(output[0], skip_special_tokens=True)

def answer_with_debug(question, pdf_path):
    lib = PDFLibrary(pdf_path)
    found_chunks = lib.find_context(question, n=1)
    context = found_chunks[0] if found_chunks else "Контекст не найден"
    
    print(f"\n[DEBUG] Retrievel find:\n{context}\n")
    print("-" * 50)

    input_text = f"Контекст: {context} Вопрос: {question}"
    inp = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    output = torch.tensor([[tokenizer.cls_token_id]]).to(DEVICE)

    for i in range(50):
        enc_mask, combined_mask, dec_mask = create_masks(inp, output)
        with torch.no_grad():
            predictions = model(inp, output, enc_mask, combined_mask, dec_mask)
        
        prediction = predictions[:, -1:, :]
        probs = torch.softmax(prediction[:, -1, :], dim=-1)
        predicted_id = torch.multinomial(probs, num_samples=1)
        output = torch.cat([output, predicted_id], dim=-1)
        
        if predicted_id == tokenizer.sep_token_id:
            break

    final_answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return final_answer

print(answer_with_debug("где отображается скорость у самоката?", "data/drivers_oth_E_elektrosamokat-kugoo-kirin-m4-pro_instrukcia_185952_07042024.pdf"))