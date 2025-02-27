import asyncio
import cv2
import os
import uuid  # для генерации коротких уникальных идентификаторов
import numpy as np
from aiogram import Bot, Dispatcher, Router
from aiogram.types import (
    FSInputFile, Message, ReplyKeyboardMarkup, KeyboardButton,
    CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
)
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from dotenv import load_dotenv
from collections import defaultdict
from ultralytics import YOLO

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(BASE_DIR, "weights", "best.pt"))

output_dir = os.path.join(BASE_DIR, "photo")
os.makedirs(output_dir, exist_ok=True)

session = AiohttpSession()
bot = Bot(token=TOKEN, session=session)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# Цены на блюда
PRICES = {
    "Biliash": 400, "Bivshteks": 450, "Jarkoe": 600, "Kasha": 400,
    "Kotleta_s_sirom": 800, "Kruasan": 400, "Kurica_i_gribi": 1000, "Latte": 180,
    "Macchoco": 180, "Maccoffe": 140, "Miaso_po_tai": 1000, "Pirojok": 350,
    "Pure": 200, "Sosiska": 150, "Sosiska_v_teste": 350, "Vareniki": 780,
    "chai": 120, "grechka": 200, "ris": 200, "snickers": 450,
    "Kotleta": 500, "Hleb": 25, "Medoviy_tortik": 600
}

# Храним выбор пользователя: user_data[user_id][dish] = count
user_data = defaultdict(lambda: defaultdict(int))

# Дополнительный кэш для распознанных блюд:
# recognized_cache[short_id] = {"items": {...}, "caption": "..."}
# где items - словарь {блюдо: количество}
recognized_cache = {}

@router.message(Command("start"))
async def handle_start(message: Message):
    kb = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/start"), KeyboardButton(text="/help"), KeyboardButton(text="/menu")]
        ],
        resize_keyboard=True
    )
    text = (
        "Привет! Я бот, который определяет блюда на подносе.\n\n"
        "Отправь мне фотографию или нажми /help."
    )
    await message.answer(text, reply_markup=kb)

@router.message(Command("help"))
async def handle_help(message: Message):
    text = (
        "Это бот для распознавания блюд на подносе.\n\n"
        "1) Отправьте фото подноса.\n"
        "2) Бот найдёт поднос, определит блюда внутри.\n"
        "3) Укажет, какие блюда обнаружены, и подсчитает итоговую стоимость.\n\n"
        "Если возникнут проблемы — [напишите администратору](youtu.be/dQw4w9WgXcQ?si=Iwzv5-ZG2HNN31E2)."
    )
    await message.answer(text, parse_mode="Markdown", disable_web_page_preview=True)

#
#  Меню (добавление блюд вручную)
#
@router.message(Command("menu"))
async def handle_menu(message: Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Добавить блюдо", callback_data="add_items"),
            InlineKeyboardButton(text="Завершить", callback_data="finish"),
        ]
    ])
    await message.answer("Выберите действие:", reply_markup=kb)


@router.callback_query(lambda c: c.data == "add_items")
async def show_items_list(callback: CallbackQuery):
    keyboard = []
    row = []
    for i, dish in enumerate(PRICES.keys(), 1):
        row.append(InlineKeyboardButton(text=dish, callback_data=f"add:{dish}"))
        # По 3 блюда в строке
        if i % 3 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    # Кнопка "Назад"
    keyboard.append([InlineKeyboardButton(text="Назад", callback_data="back_to_menu")])

    kb = InlineKeyboardMarkup(inline_keyboard=keyboard)
    await callback.message.edit_text("Нажмите на блюдо, чтобы добавить 1 шт.", reply_markup=kb)


@router.callback_query(lambda c: c.data and c.data.startswith("add:"))
async def add_item_callback(callback: CallbackQuery):
    _, dish = callback.data.split(":")
    user_id = callback.from_user.id
    user_data[user_id][dish] += 1
    await callback.answer(f"Добавлено: {dish}", show_alert=False)


@router.callback_query(lambda c: c.data == "back_to_menu")
async def back_to_menu_callback(callback: CallbackQuery):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Добавить блюдо", callback_data="add_items"),
            InlineKeyboardButton(text="Завершить", callback_data="finish"),
        ]
    ])
    await callback.message.edit_text("Выберите действие:", reply_markup=kb)


@router.callback_query(lambda c: c.data == "finish")
async def finish_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    items = user_data[user_id]
    if not items:
        await callback.message.edit_text("Вы не выбрали ни одного блюда.")
        return

    lines = []
    total_price = 0
    for dish, qty in items.items():
        price = PRICES[dish] * qty
        total_price += price
        lines.append(f"{dish} x{qty} = {price} KZT")

    summary = "\n".join(lines)
    summary += f"\n\nИтого: {total_price} KZT"

    # Очищаем, если всё завершаем
    user_data[user_id].clear()

    await callback.message.edit_text(f"Ваш заказ:\n\n{summary}")

#
#  Функция для распознавания блюд на подносе
#
async def process_image(image_path):
    results = model(image_path)
    img = cv2.imread(image_path)

    detected_items = {}
    tray_bbox = None
    max_area = 0

    # 1) Ищем самый большой "podnos"
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            if model.names[class_id].lower() == "podnos":
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    tray_bbox = (x1, y1, x2, y2)

    if tray_bbox:
        x1, y1, x2, y2 = tray_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(img, "Podnos", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Обрезаем
        crop_img = img[y1:y2, x1:x2]

        # Смотрим блюда внутри
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                item_name = model.names[class_id]
                conf = box.conf[0] * 100

                if item_name.lower() == "podnos":
                    continue
                if bx1 >= x1 and bx2 <= x2 and by1 >= y1 and by2 <= y2:
                    detected_items[item_name] = detected_items.get(item_name, 0) + 1
                    cv2.rectangle(crop_img, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), (0, 255, 0), 3)
                    label = f"{item_name} {conf:.1f}%"
                    cv2.putText(crop_img, label, (bx1 - x1, by1 - y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img = crop_img

    # Сохраняем итоговое изображение
    output_path = os.path.join(output_dir, "result.jpg")
    cv2.imwrite(output_path, img)

    # Считаем сумму
    total_price = 0
    items_text = []
    for item, count in detected_items.items():
        price = PRICES.get(item, 0) * count
        total_price += price
        items_text.append(f"{item} x{count} — {price} KZT")

    caption = "\n".join(items_text) + f"\n\n💰 **Итог: {total_price} KZT**" if items_text else "Еда не найдена"
    return output_path, detected_items, caption

#
#  Хендлер для фото
#
@router.message(lambda message: message.photo)
async def handle_photo(message: Message):
    # Сохраняем фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    img_path = os.path.join(output_dir, f"{photo.file_id}.jpg")
    await bot.download_file(file.file_path, img_path)

    # Вызываем process_image
    output_path, recognized_items, caption = await process_image(img_path)

    # Показываем итоговое фото
    photo_file = FSInputFile(output_path)
    await message.answer_photo(photo_file, caption=caption)

    # Генерируем короткий идентификатор (например, 8 символов из uuid)
    short_id = str(uuid.uuid4())[:8]
    recognized_cache[short_id] = {
        "items": recognized_items,
        "caption": caption
    }

    # Если есть блюда, предлагаем добавить к заказу или сохранить отдельно
    if recognized_items:
        kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(
                text="Добавить к списку",
                callback_data=f"add_from_photo:{short_id}"
            ),
            InlineKeyboardButton(
                text="Записать отдельно",
                callback_data=f"save_separate:{short_id}"
            )
        ]])
        await message.answer(
            text="Хотите добавить новые блюда к текущему заказу или записать отдельно?",
            reply_markup=kb
        )
    else:
        await message.answer("На подносе не найдено блюд. Попробуйте другое фото.")


#
#  Обработка "Добавить к списку"
#
@router.callback_query(lambda c: c.data and c.data.startswith("add_from_photo:"))
async def add_from_photo_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    _, short_id = callback.data.split(":")
    # Достаём из кэша
    data = recognized_cache.get(short_id)
    if not data:
        await callback.message.edit_text("Данные не найдены (истёк кэш).")
        return

    recognized_items = data["items"]
    if recognized_items:
        # Прибавляем к заказу
        for dish, qty in recognized_items.items():
            user_data[user_id][dish] += qty

        await callback.message.edit_text(
            "Добавлено к заказу!\n\n"
            "Чтобы вручную добавить ещё блюда, используйте /menu.\n"
            "Или завершите, нажав 'Завершить' в меню."
        )
    else:
        await callback.message.edit_text("Ничего не обнаружено. Заказ не изменён.")

    # По желанию можно очистить запись в кэше
    recognized_cache.pop(short_id, None)

#
#  Обработка "Записать отдельно"
#
@router.callback_query(lambda c: c.data and c.data.startswith("save_separate:"))
async def save_separate_callback(callback: CallbackQuery):
    _, short_id = callback.data.split(":")
    data = recognized_cache.get(short_id)
    if not data:
        await callback.message.edit_text("Данные не найдены (истёк кэш).")
        return

    caption = data["caption"]
    recognized_items = data["items"]

    if not recognized_items:
        await callback.message.edit_text("Ничего не было обнаружено.")
    else:
        await callback.message.edit_text(
            "Отдельный заказ зафиксирован:\n\n" + caption
        )

    # По желанию удалить из кэша
    recognized_cache.pop(short_id, None)

#
#  Запуск
#
async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
