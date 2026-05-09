import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Количество GPU: {torch.cuda.device_count()}")
print(f"Имя карты: {torch.cuda.get_device_name(0)}")