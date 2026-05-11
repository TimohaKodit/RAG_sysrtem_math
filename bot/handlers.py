import aiogram 
from aiogram import Router, types, F
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from bot.state.state_query import Query_text
from aiogram.types import Message, CallbackQuery, ContentType
import aiohttp
import asyncio
from api import Query
from .keyboard import load_file
from ingest import dc

handler_rout = Router()


@handler_rout.message(Command("start"))
async def message_hello(message: types.Message, state: FSMContext):
    await message.answer("Привет, напиши запрос который хочешь узнать!", reply_markup=await load_file())
    await state.clear()
    
    



@handler_rout.message(F.text)
async def get_query(message: Message, state: FSMContext):
    chat = (await state.get_data()).get('chat_history', [])
    
    payload = {
        "input": message.text,
        "chat_history": chat,
        "user_id": message.from_user.id  # Передаем ID пользователя ботом
    }
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:8000/ask', json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                new_user_msg = {"role": "user", "content": message.text}
                new_ai_msg = {"role": "ai", "content": result['answer']}
                await message.answer(result['answer'])
                chat.append(new_user_msg)
                chat.append(new_ai_msg)

                chat = chat[-10:]
                await state.update_data(chat_history=chat)
    
    

@handler_rout.callback_query(F.data=='load')
async def file(callback: CallbackQuery):
    await callback.answer("Загрузи файл")

@handler_rout.message(F.document)
async def get_file(message: Message, bot: aiogram.Bot):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    
    file_path_directory =f"data/{message.document.file_name}"

    await bot.download_file(file_path, destination=f"data/{message.document.file_name}")
    await asyncio.to_thread(dc, file_path_directory, message.from_user.id)
    await message.answer("Файл успешно сохранен!")