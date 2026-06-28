# -*- coding: utf-8 -*-
"""
Задание 17 - Бабенко
Сжатие изображения с помощью усечения спектра (2D БПФ)

Алгоритм:
1. Вычислить двумерное БПФ изображения.
2. Отсортировать амплитуды всех частотных коэффициентов по убыванию.
3. Оставить только N% коэффициентов с наибольшей амплитудой, остальные обнулить.
4. Выполнить обратное БПФ -> восстановленное изображение.
5. Оценить качество через PSNR/MSE, построить график PSNR(N%).

Дополнительно: сохранение сжатого изображения как списка индексов и значений
оставшихся коэффициентов + последующая загрузка и восстановление.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import os

np.random.seed(0)

# ---------------------------------------------------------------------------
# 0. Подготовка тестового изображения (градации серого)
# ---------------------------------------------------------------------------

def generate_test_image(size=256):
    """
    Генерирует синтетическое тестовое изображение в градациях серого
    с резкими границами, мелкими деталями и небольшим шумом - такое
    изображение хорошо иллюстрирует эффект усечения спектра Фурье
    (на гладких изображениях разница почти не заметна даже при N=1%).
    """
    rng = np.random.RandomState(42)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    img = np.zeros((size, size), dtype=np.float64)

    # Шахматная доска (резкие высокочастотные границы)
    cell = size // 16
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                img[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = 200

    # Светлый круг в центре (низкочастотный плавный объект)
    R = np.sqrt(X**2 + Y**2)
    circle_mask = R < 0.25
    img[circle_mask] = 60

    # Несколько резких прямых линий (высокие частоты)
    img[size // 2 - 1: size // 2 + 1, :] = 255
    img[:, size // 4 - 1: size // 4 + 1] = 0

    # Диагональная линия
    for k in range(size):
        if 0 <= k < size:
            img[k, k] = 255 if k < size else img[k, k]

    # Небольшой гауссов шум (добавляет высокочастотные компоненты)
    noise = rng.normal(0, 8, (size, size))
    img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


IMG_PATH = "/home/claude/test_image.png"
image = generate_test_image(256)
Image.fromarray(image).save(IMG_PATH)
print(f"Тестовое изображение сгенерировано: {image.shape}, сохранено в {IMG_PATH}")

# Если у вас есть собственное изображение, замените блок выше на:
#   image = np.array(Image.open("ваш_файл.png").convert("L"))


# ---------------------------------------------------------------------------
# 1-3. Сжатие: БПФ -> сортировка по амплитуде -> оставить top-N% -> обнулить остальное
# ---------------------------------------------------------------------------

def compress_by_spectrum_truncation(img, percent):
    """
    Сжимает изображение, оставляя только percent% коэффициентов Фурье
    с наибольшей амплитудой.

    Параметры
    ---------
    img : 2D ndarray (градации серого)
    percent : float, доля коэффициентов для сохранения (0 < percent <= 100)

    Возвращает
    ----------
    reconstructed : 2D ndarray (uint8) - восстановленное изображение
    fft_truncated : 2D ndarray (complex) - усечённый спектр (для отладки/доп. задания)
    """
    img_f = img.astype(np.float64)

    # 1. Двумерное БПФ
    F = np.fft.fft2(img_f)

    # 2. Сортировка амплитуд по убыванию (работаем с "сплющенным" массивом)
    magnitude = np.abs(F)
    flat_mag = magnitude.flatten()
    # Индексы, которые отсортировали бы amplitude по убыванию.
    # Используем argsort по возрастанию и берём порог через kth-элемент - быстрее для больших N.
    n_total = flat_mag.size
    n_keep = max(1, int(np.round(n_total * percent / 100.0)))

    # Порог амплитуды: n_keep-й по величине элемент (быстрее, чем полная сортировка)
    threshold_idx = n_total - n_keep
    partitioned = np.partition(flat_mag, threshold_idx)
    threshold_value = partitioned[threshold_idx]

    # 3. Маска: оставляем коэффициенты с амплитудой >= порога, остальные - 0
    mask = magnitude >= threshold_value
    # На случай, если из-за повторяющихся значений порога коэффициентов чуть больше n_keep -
    # это не страшно, поведение детерминировано и стандартно для таких задач.

    F_truncated = F * mask

    # 4. Обратное БПФ
    img_reconstructed = np.fft.ifft2(F_truncated)
    img_reconstructed = np.real(img_reconstructed)
    img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)

    return img_reconstructed, F_truncated, mask


# ---------------------------------------------------------------------------
# 5. Метрики качества: MSE и PSNR
# ---------------------------------------------------------------------------

def compute_mse(original, reconstructed):
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    return np.mean((original - reconstructed) ** 2)


def compute_psnr(original, reconstructed, max_pixel=255.0):
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# ---------------------------------------------------------------------------
# Демонстрация для нескольких процентов (10%, 5%, 1% + пара дополнительных)
# ---------------------------------------------------------------------------

percents_to_show = [50, 10, 5, 1]
results = {}

for p in percents_to_show:
    rec, F_trunc, mask = compress_by_spectrum_truncation(image, p)
    mse = compute_mse(image, rec)
    psnr = compute_psnr(image, rec)
    results[p] = {"reconstructed": rec, "mse": mse, "psnr": psnr}
    print(f"N = {p:>5}% | MSE = {mse:10.3f} | PSNR = {psnr:6.2f} dB")

# Визуализация: оригинал + восстановленные изображения для разных N%
fig, axes = plt.subplots(1, len(percents_to_show) + 1, figsize=(4 * (len(percents_to_show) + 1), 4))

axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Оригинал")
axes[0].axis('off')

for ax, p in zip(axes[1:], percents_to_show):
    ax.imshow(results[p]["reconstructed"], cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"N={p}%\nPSNR={results[p]['psnr']:.1f} дБ")
    ax.axis('off')

plt.tight_layout()
plt.savefig("/home/claude/comparison.png", dpi=120)
plt.close()
print("Сохранено сравнение изображений: comparison.png")


# ---------------------------------------------------------------------------
# 6. График зависимости PSNR от процента оставленных коэффициентов
# ---------------------------------------------------------------------------

percent_range = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100])
psnr_values = []
mse_values = []

for p in percent_range:
    rec, _, _ = compress_by_spectrum_truncation(image, p)
    psnr_values.append(compute_psnr(image, rec))
    mse_values.append(compute_mse(image, rec))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(percent_range, psnr_values, marker='o', color='tab:blue')
ax1.set_xlabel("Процент оставленных коэффициентов, N (%)")
ax1.set_ylabel("PSNR, дБ")
ax1.set_title("Зависимость PSNR от N%")
ax1.grid(True, alpha=0.3)

ax2.plot(percent_range, mse_values, marker='s', color='tab:red')
ax2.set_xlabel("Процент оставленных коэффициентов, N (%)")
ax2.set_ylabel("MSE")
ax2.set_title("Зависимость MSE от N%")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/psnr_vs_percent.png", dpi=120)
plt.close()
print("Сохранён график PSNR(N%) и MSE(N%): psnr_vs_percent.png")


# ---------------------------------------------------------------------------
# ДОПОЛНИТЕЛЬНО: сохранение сжатого представления (индексы + значения)
# и последующая загрузка/восстановление
# ---------------------------------------------------------------------------

def save_compressed(F_truncated, mask, shape, filepath):
    """
    Сохраняет сжатое представление спектра как список индексов
    ненулевых коэффициентов и их комплексных значений (а не весь массив).
    """
    nonzero_indices = np.nonzero(mask)  # tuple (rows, cols)
    rows, cols = nonzero_indices
    values = F_truncated[rows, cols]  # комплексные значения коэффициентов

    data = {
        "shape": shape,
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "values": values.astype(np.complex128),
    }

    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    # Сравнение размеров для иллюстрации эффективности сжатия
    full_size = shape[0] * shape[1] * np.dtype(np.complex128).itemsize
    sparse_size = (
        rows.nbytes + cols.nbytes + values.nbytes
    )
    return full_size, sparse_size


def load_and_reconstruct(filepath):
    """
    Загружает сжатое представление и восстанавливает изображение
    через обратное БПФ.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    shape = data["shape"]
    rows, cols, values = data["rows"], data["cols"], data["values"]

    F_restored = np.zeros(shape, dtype=np.complex128)
    F_restored[rows, cols] = values

    img_restored = np.fft.ifft2(F_restored)
    img_restored = np.clip(np.real(img_restored), 0, 255).astype(np.uint8)
    return img_restored


# Демонстрация для N = 10%
demo_percent = 10
rec_demo, F_trunc_demo, mask_demo = compress_by_spectrum_truncation(image, demo_percent)

COMPRESSED_PATH = "/home/claude/compressed_data.pkl"
full_bytes, sparse_bytes = save_compressed(F_trunc_demo, mask_demo, image.shape, COMPRESSED_PATH)

print(f"\n--- Дополнительное задание (N={demo_percent}%) ---")
print(f"Полный спектр (комплексный, float64): {full_bytes:,} байт")
print(f"Сжатое представление (индексы + значения): {sparse_bytes:,} байт")
print(f"Коэффициент сжатия: {full_bytes / sparse_bytes:.2f}x")

img_loaded = load_and_reconstruct(COMPRESSED_PATH)

# Проверка, что восстановление из файла совпадает с прямым восстановлением
assert np.array_equal(img_loaded, rec_demo), "Восстановленное из файла изображение не совпадает!"
print("Восстановление из сохранённого файла совпадает с прямым восстановлением: OK")

psnr_loaded = compute_psnr(image, img_loaded)
print(f"PSNR восстановленного из файла изображения: {psnr_loaded:.2f} дБ")

print("\nВсе файлы готовы: test_image.png, comparison.png, psnr_vs_percent.png, compressed_data.pkl")
