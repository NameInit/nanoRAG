## nanoRAG 🧬
nanoRAG — это легковесная и модульная реализация классической архитектуры Transformer на PyTorch. Проект предназначен для изучения механизмов работы Attention, Encoder-Decoder связок и последующего использования в задачах Retrieval-Augmented Generation (RAG).
## ✨ Особенности

* Чистый PyTorch: Реализация «с нуля» без использования тяжелых библиотек вроде Hugging Face (только torch и nn.Module).
* Классическая архитектура: Полноценный Encoder и Decoder с механизмом Multi-Head Attention.
* Модульность: Каждый компонент (Attention, LayerNorm, Positional Encoding) выделен в отдельный блок для удобства кастомизации.
* Готовность к RAG: Структура проекта оптимизирована под пайплайн «поиск + генерация».

## 🏗 Архитектура модели
Реализация следует каноничному дизайну "Attention Is All You Need":

   1. Encoder: Обрабатывает входную последовательность (контекст/найденные документы).
   2. Decoder: Генерирует ответ, используя Cross-Attention на выходы энкодера.
   3. Multi-Head Attention: Позволяет модели фокусироваться на разных частях последовательности одновременно.
   4. Positional Encoding: Добавляет информацию о порядке слов в эмбеддинги.

## 📂 Структура проекта

nanoRAG/
├── src/
│   ├── encoder.py              # Слоевая реализация энкодера
│   ├── decoder.py              # Слоевая реализация декодера
│   ├── transformer.py          # Объединение блоков в единую модель
│   ├── multi_head_attention.py # Механизм внимания
│   ├── loader.py               # Загрузка и подготовка данных
│   └── utils.py                # Позиционное кодирование и хелперы
├── train.py                    # Основной цикл обучения
├── finetune.py                 # Скрипт для дообучения на специфичных данных
├── inference.py                # Запуск модели для генерации
└── requirements.txt            # Список зависимостей

## 🚀 Быстрый старт
## 1. Установка

git clone https://github.com/NameInit/nanoRAG.git

cd nanoRAG

pip install -r requirements.txt

## 2. Обучение
Для запуска базового обучения выполните:

python train.py

## 3. Инференс (Генерация)
Используйте веса обученной модели для получения ответов:

python inference.py --prompt "Текст вашего запроса"

## 🛠 Технологический стек

* Framework: PyTorch
* Language: Python 3.8+
* Concepts: Transformers, Multi-Head Attention, RAG, NLP

