"""CRNN-модель для распознавания текста ГРЗ.

Архитектура:
  CNN (feature extractor) → BiLSTM → CTC-head

Вход:  grayscale изображение 32x128 (H x W)
Выход: последовательность символов (CTC)
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN: CNN + BiLSTM + CTC.

    Вход: (B, 1, 32, W) — batch, 1 канал (grayscale), высота 32, ширина W
    Выход: (T, B, num_classes) — T временных шагов, B batch, classes
    """

    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.num_classes = num_classes

        # -------- CNN --------
        # Вход: (B, 1, 32, W)
        # После каждого MaxPool высота делится на 2, ширина тоже
        self.cnn = nn.Sequential(
            # -> (B, 64, 16, W/2)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # -> (B, 128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # -> (B, 256, 4, W/8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # -> (B, 256, 2, W/16)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # высота 2->1, ширина не меняется

            # -> (B, 512, 1, W/16)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # высота 1->... ок, округление, но с padding=1 высота 1 не делится
        )

        # Вместо последнего пула — оставим высоту 1
        # Перепишем чуть проще:

        # Высота входа 32 -> после 3 макспулов высота 4.
        # Мы хотим высоту 1. Значит нужно 5 макспулов по высоте.
        # Сделаем проще: после всех conv делаем adaptive avg pool по высоте

        # (оставлю выше как есть и добавлю adaptive avg pool ниже)

        self.cnn_out_height = 1

        # -------- BiLSTM --------
        self.rnn = nn.LSTM(
            input_size=512,   # совпадает с последним Conv2d каналов
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,  # вход (T, B, F)
            dropout=0.1,
        )

        # -------- CTC head --------
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        x: (B, 1, 32, W)
        Возвращает: (T, B, num_classes) — log-probs для CTC
        """
        # CNN
        conv = self.cnn(x)                    # (B, 512, H', W')
        # Усредняем по высоте -> (B, 512, W')
        conv = conv.mean(dim=2)               # (B, 512, W')

        # Готовим для LSTM: (B, C, W') -> (W', B, C)
        conv = conv.permute(2, 0, 1)          # (W', B, 512)

        # BiLSTM
        rnn_out, _ = self.rnn(conv)           # (W', B, 2*hidden)

        # CTC head
        logits = self.fc(rnn_out)             # (W', B, num_classes)
        log_probs = logits.log_softmax(dim=2) # (W', B, num_classes)

        return log_probs


def ctc_decode(log_probs, blank=0):
    """Жадное CTC-декодирование.
    log_probs: (T, B, C) — numpy или torch
    Возвращает: список строк (для каждого элемента батча)
    """
    if isinstance(log_probs, torch.Tensor):
        log_probs = log_probs.detach().cpu().numpy()

    import numpy as np

    # Индексы максимальных вероятностей на каждом шаге
    preds = log_probs.argmax(axis=2)  # (T, B)
    T, B = preds.shape

    results = []
    for b in range(B):
        seq = preds[:, b]
        # Убираем повторы и blank
        out = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                out.append(int(idx))
            prev = idx
        results.append(out)
    return results


if __name__ == "__main__":
    # Тест
    model = CRNN(num_classes=23)  # 22 символа + blank
    x = torch.randn(2, 1, 32, 128)  # batch=2, 1 канал, H=32, W=128
    out = model(x)
    print(f"Вход: {x.shape}")
    print(f"Выход: {out.shape}")  # ожидаем (T=32, B=2, C=23)
    print("CRNN работает.")