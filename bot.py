import asyncio
import cv2
import os
import numpy as np
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import FSInputFile, Message
from aiogram.client.session.aiohttp import AiohttpSession
from ultralytics import YOLO

# 🔒 Укажи свой токен Telegram-бота
TOKEN = "7219854065:AAF9HPxkzr63qGpV2KRg3AH1UtkT7g9FdkU"

# 🔥 Загружаем обученную модель YOLO
model = YOLO(r"C:\Users\Rudeus\Desktop\Projekttt\my_experiment2\weights\best.pt")

# 📂 Папка для хранения фото
output_dir = r"C:\Users\Rudeus\Desktop\photo"
os.makedirs(output_dir, exist_ok=True)

# 💰 Обновленные цены на блюда (в тенге)
PRICES = {
    "Biliash": 400, "Bivshteks": 450, "Jarkoe": 600, "Kasha": 400,
    "Kotleta_s_sirom": 800, "Kruasan": 400, "Kurica_i_gribi": 1000, "Latte": 180,
    "Macchoco": 180, "Maccoffe": 140, "Miaso_po_tai": 1000, "Pirojok": 350,
    "Pure": 200, "Sosiska": 150, "Sosiska_v_teste": 350, "Vareniki": 780,
    "chai": 120, "grechka": 200, "ris": 200, "snickers": 450
}

# 🔧 Создаем бота и диспетчер
session = AiohttpSession()
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()
router = Router()
dp.include_router(router)


async def process_image(image_path):
    """Обрабатывает изображение через YOLO, формирует список еды, считает цену и обрезает по подносу"""
    results = model(image_path)
    img = cv2.imread(image_path)

    detected_items = {}  # Словарь для хранения количества каждого блюда
    tray_bbox = None  # Координаты подноса

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0] * 100  # Переводим в проценты
            label = f"{model.names[class_id]} {confidence:.1f}%"  # Название + уверенность

            # Рисуем рамки и подписи на изображении
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Если найден поднос – запоминаем координаты
            if model.names[class_id].lower() == "podnos":
                tray_bbox = (x1, y1, x2, y2)

            # Добавляем в список (без подноса)
            else:
                item_name = model.names[class_id]
                detected_items[item_name] = detected_items.get(item_name, 0) + 1

    # Если поднос найден – обрезаем изображение
    if tray_bbox:
        x1, y1, x2, y2 = tray_bbox
        img = img[y1:y2, x1:x2]

    # Сохраняем изображение
    output_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(output_path, img)

    # Формируем список блюд и считаем общую стоимость
    total_price = 0
    items_text = []

    for item, count in detected_items.items():
        price = PRICES.get(item, 0) * count
        total_price += price
        items_text.append(f"{item} x{count} — {price} KZT")

    # Итоговый текст для пользователя
    caption = "\n".join(items_text) + f"\n\n💰 **Итог: {total_price} KZT**" if items_text else "Еда не найдена"
    return output_path, caption


@router.message(lambda message: message.photo)
async def handle_photo(message: Message):
    """Обрабатывает фото, отправленное пользователем"""
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_path = os.path.join(output_dir, f"{photo.file_id}.jpg")

    await bot.download_file(file.file_path, img_path)

    output_path, caption = await process_image(img_path)  # Получаем фото + подпись

    # ✅ Отправляем результат с подписью
    photo_file = FSInputFile(output_path)
    await message.answer_photo(photo_file, caption=caption)


async def main():
    """Запуск бота"""
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
