from aiogram import Dispatcher, Bot
import os
import logging
import asyncio
from bot.handlers import handler_rout

from dotenv import load_dotenv

load_dotenv()

async def main():

    bot = Bot(token=os.getenv("BOT_API_KEY"))

    dp = Dispatcher()
    dp.include_router(handler_rout)

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())