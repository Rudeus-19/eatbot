import os
import asyncio
import cv2
import numpy as np
import requests
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import FSInputFile, Message
from aiogram.client.session.aiohttp import AiohttpSession
from ultralytics import YOLO
from dotenv import load_dotenv

# ✅ Загружаем переменные окружения (если есть .env)
if os.path.exists(".env"):
    load_dotenv()

# ✅ Берём токен из .env (локально) или из Render (на сервере)
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError("❌ Токен не найден! Убедись, что BOT_TOKEN задан в .env или в Render!")

# ✅ Определяем путь к проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best.pt")

# ✅ Создаём папку для модели, если её нет
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ✅ Если модели нет – скачиваем (если она весит >100MB и не в GitHub)
MODEL_URL = "https://your-cloud.com/path-to-best.pt"  # Укажи ссылку на скачивание

if not os.path.exists(MODEL_PATH):
    print("🔄 Загружаем модель YOLO...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Модель загружена!")

# ✅ Загружаем YOLO
model = YOLO(MODEL_PATH)

# ✅ Создаём папку для фото (если её нет)
output_dir = os.path.join(BASE_DIR, "photo")
os.makedirs(output_dir, exist_ok=True)

# 🔧 Создаём бота и диспетчер
session = AiohttpSession()
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# 💰 Цены на блюда (в тенге)
PRICES = {
    "Biliash": 400, "Bivshteks": 450, "Jarkoe": 600, "Kasha": 400,
    "Kotleta_s_sirom": 800, "Kruasan": 400, "Kurica_i_gribi": 1000, "Latte": 180,
    "Macchoco": 180, "Maccoffe": 140, "Miaso_po_tai": 1000, "Pirojok": 350,
    "Pure": 200, "Sosiska": 150, "Sosiska_v_teste": 350, "Vareniki": 780,
    "chai": 120, "grechka": 200, "ris": 200, "snickers": 450,
    "Kotleta": 500, "Hleb": 25, "Medoviy_tortik": 600
}

async def process_image(image_path):
    """Обрабатывает изображение через YOLO, выделяет самый большой поднос и блюда на нём"""
    results = model(image_path)
    img = cv2.imread(image_path)

    detected_items = {}  # Словарь для хранения количества каждого блюда
    tray_bbox = None  # Координаты самого большого подноса
    max_area = 0  # Переменная для хранения максимальной площади

    # 1️⃣ Поиск самого большого подноса
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            # Если найден поднос, выбираем самый большой
            if model.names[class_id].lower() == "podnos":
                area = (x2 - x1) * (y2 - y1)  # Площадь подноса
                if area > max_area:
                    max_area = area
                    tray_bbox = (x1, y1, x2, y2)

    # 2️⃣ Если найден поднос, фильтруем блюда внутри него и обводим поднос
    if tray_bbox:
        x1, y1, x2, y2 = tray_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(img, "Podnos", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        img = img[y1:y2, x1:x2]

        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                item_name = model.names[class_id]
                confidence = box.conf[0] * 100

                if item_name.lower() == "podnos":
                    continue  

                if bx1 >= x1 and bx2 <= x2 and by1 >= y1 and by2 <= y2:
                    detected_items[item_name] = detected_items.get(item_name, 0) + 1
                    cv2.rectangle(img, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (0, 255, 0), 3)
                    label = f"{item_name} {confidence:.1f}%"
                    cv2.putText(img, label, (bx1 - x1, by1 - y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(output_path, img)

    total_price = 0
    items_text = []

    for item, count in detected_items.items():
        price = PRICES.get(item, 0) * count
        total_price += price
        items_text.append(f"{item} x{count} — {price} KZT")

    caption = "\n".join(items_text) + f"\n\n💰 **Итог: {total_price} KZT**" if items_text else "Еда не найдена"
    return output_path, caption

@router.message(lambda message: message.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_path = os.path.join(output_dir, f"{photo.file_id}.jpg")
    await bot.download_file(file.file_path, img_path)

    output_path, caption = await process_image(img_path)
    photo_file = FSInputFile(output_path)
    await message.answer_photo(photo_file, caption=caption)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
