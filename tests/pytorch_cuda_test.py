import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision.models import resnet50
if not torch.cuda.is_available():
    raise RuntimeError("CUDA не доступна!")

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}")
torch.backends.cudnn.benchmark = True

# === Параметры нагрузки ===
batch_size = 128        # можно увеличить до 256/512, если память позволяет
img_size = 224
in_channels = 3
num_classes = 1000
steps = 500             # ← увеличь для большей нагрузки (500–2000 шагов)
use_amp = True          # mixed precision — увеличивает нагрузку на Tensor Cores

# === Более тяжёлая модель ===
model = resnet50(weights=None, num_classes=num_classes).cuda()
model = model.to(device)

# === Синтетические данные (остаются на GPU) ===
X = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
y = torch.randint(0, num_classes, (batch_size,), device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

print(f"\nЗапуск нагрузочного теста: {steps} шагов на ResNet-50 (batch={batch_size})")
print("Следи за загрузкой через `nvidia-smi` или `nvtop`...\n")

model.train()
torch.cuda.synchronize()  # точный замер времени
start_time = time.time()

for step in range(steps):
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=use_amp):
        output = model(X)
        loss = criterion(output, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        print(f"Шаг [{step}/{steps}], Loss: {loss.item():.4f}")

torch.cuda.synchronize()
elapsed = time.time() - start_time

print(f"\n✅ Нагрузочный тест завершён за {elapsed:.2f} сек.")
print(f"Средняя скорость: {steps / elapsed:.1f} шагов/сек")