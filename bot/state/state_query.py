from aiogram.fsm.state import State, StatesGroup

class Query_text(StatesGroup):
    query = State()
    chat_history = State()
    

