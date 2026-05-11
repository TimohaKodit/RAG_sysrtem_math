from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton



async def load_file() -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup(
        inline_keyboard = [
            [InlineKeyboardButton(text="Загрузить", callback_data='load')]
        ]
    )

    return keyboard