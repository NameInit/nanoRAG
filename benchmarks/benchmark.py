import json
import torch
import re
import sys
import os
from typing import Dict, List
from transformers import AutoTokenizer
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PDFLibrary, Transformer, create_masks, DEVICE

class SimpleBenchmark:
    def __init__(self, qa_json_path: str, pdf_path: str, model_path: str):
        """Загружаем тестовые вопросы и модель"""
        with open(qa_json_path, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        
        self.pdf_lib = PDFLibrary(pdf_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.model = Transformer(
            num_layers=4, d_model=256, num_heads=8, dff=512,
            input_vocab_size=self.tokenizer.vocab_size,
            target_vocab_size=self.tokenizer.vocab_size,
            max_pos=512
        ).to(DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
    
    def normalize(self, text: str) -> str:
        """Приводим текст к единому виду"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())
    
    def exact_match(self, pred: str, true: str) -> bool:
        """Точное совпадение"""
        return self.normalize(pred) == self.normalize(true)
    
    def f1_score(self, pred: str, true: str) -> float:
        """F1 метрика"""
        pred_tokens = self.normalize(pred).split()
        true_tokens = self.normalize(true).split()
        
        if not pred_tokens and not true_tokens:
            return 1.0
        if not pred_tokens or not true_tokens:
            return 0.0
        
        common = set(pred_tokens) & set(true_tokens)
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(true_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def generate_answer(self, question: str) -> str:
        """Генерируем ответ моделью"""
        chunks = self.pdf_lib.find_context(question, n=2)
        context = " ".join(chunks[:2])
        
        input_text = f"Контекст: {context} Вопрос: {question}"
        
        inp = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512,
            truncation=True
        ).to(DEVICE)
        
        start_token = self.tokenizer.sep_token_id
        generated = [start_token]
        
        for _ in range(50):
            output_tensor = torch.tensor([generated], dtype=torch.long, device=DEVICE)
            
            if output_tensor.size(1) > 512:
                output_tensor = output_tensor[:, -512:]
            
            enc_mask, combined_mask, dec_mask = create_masks(inp, output_tensor)
            
            with torch.no_grad():
                predictions = self.model(inp, output_tensor, enc_mask, combined_mask, dec_mask)
            
            next_token = torch.argmax(predictions[:, -1, :], dim=-1).item()
            
            if next_token == self.tokenizer.sep_token_id:
                break
            
            generated.append(next_token)
            
            if len(generated) > 100:
                break
        
        answer = self.tokenizer.decode(generated[1:], skip_special_tokens=True)
        return answer if answer.strip() else "[НЕТ ОТВЕТА]"
    
    def run(self):
        """Запускаем тестирование"""
        print(f"\n{'='*50}")
        print(f"Тестирование модели на {len(self.qa_pairs)} вопросах")
        print(f"{'='*50}\n")
        
        exact_matches = 0
        f1_scores = []
        empty_answers = 0
        
        for i, qa in enumerate(tqdm(self.qa_pairs, desc="Тестирование")):
            try:
                pred = self.generate_answer(qa['question'])
                true = qa['answer']
                
                if pred == "[НЕТ ОТВЕТА]":
                    empty_answers += 1
                
                if self.exact_match(pred, true):
                    exact_matches += 1
                
                f1 = self.f1_score(pred, true)
                f1_scores.append(f1)
                
                if i < 5:
                    print(f"\n--- Вопрос {i+1} ---")
                    print(f"Q: {qa['question']}")
                    print(f"True: {true[:80]}...")
                    print(f"Pred: {pred[:80]}...")
                    print(f"F1: {f1:.3f}")
            except Exception as e:
                print(f"\nОшибка на вопросе {i+1}: {e}")
                f1_scores.append(0.0)
        
        accuracy = (exact_matches / len(self.qa_pairs)) * 100
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        print(f"\n{'='*50}")
        print("РЕЗУЛЬТАТЫ")
        print(f"{'='*50}")
        print(f"Exact Match Accuracy: {accuracy:.2f}%")
        print(f"Средний F1 Score: {avg_f1:.3f}")
        print(f"Правильных ответов: {exact_matches}/{len(self.qa_pairs)}")
        print(f"Пустых ответов: {empty_answers}/{len(self.qa_pairs)}")
        
        if avg_f1 >= 0.8:
            quality = "Отлично!"
        elif avg_f1 >= 0.6:
            quality = "Хорошо"
        elif avg_f1 >= 0.4:
            quality = "Средне"
        else:
            quality = "Плохо (нужно дообучение)"
        
        print(f"Качество модели: {quality}")
        print(f"{'='*50}")
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_file = f"benchmark_results_{timestamp}.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: Custom Transformer (Kugoo)\n")
            f.write(f"Test questions: {len(self.qa_pairs)}\n")
            f.write(f"Exact Match Accuracy: {accuracy:.2f}%\n")
            f.write(f"Average F1 Score: {avg_f1:.3f}\n")
            f.write(f"Correct: {exact_matches}/{len(self.qa_pairs)}\n")
            f.write(f"Empty answers: {empty_answers}/{len(self.qa_pairs)}\n")
            f.write(f"Quality: {quality}\n")
        
        print(f"\nРезультаты сохранены в: {results_file}")
        
        return {'accuracy': accuracy, 'avg_f1': avg_f1}

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    benchmark = SimpleBenchmark(
        qa_json_path=os.path.join(current_dir, "../data/kugoo_faq.json"),
        pdf_path=os.path.join(current_dir, "../data/drivers_oth_E_elektrosamokat-kugoo-kirin-m4-pro_instrukcia_185952_07042024.pdf"),
        model_path=os.path.join(current_dir, "../model/ft_transformer_20260508-1045_ep14_loss_2.08.pth")
    )
    
    results = benchmark.run()