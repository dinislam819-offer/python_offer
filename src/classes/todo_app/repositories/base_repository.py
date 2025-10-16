import json
from datetime import datetime
from json import JSONEncoder
from typing import TypeVar, Generic, Dict, Optional, Type, List
from pydantic import BaseModel
import os

T = TypeVar('T', bound=BaseModel)

class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class BaseRepository(Generic[T]):
    def __init__(self, file_path: str, model: Type[T]):
        # TODO: Реализуйте инициализацию
        pass

    def _load(self) -> None:
        # TODO: Реализуйте загрузку данных из файла
        pass

    def _save(self) -> None:
        # TODO: Реализуйте сохранение данных в файл
        pass

    def add(self, item: T) -> T:
        # TODO: Реализуйте добавление элемента
        pass

    def get(self, id_: int) -> Optional[T]:
        # TODO: Реализуйте получение элемента по ID
        pass

    def get_all(self) -> List[T]:
        # TODO: Реализуйте получение всех элементов
        pass

    def update(self, id_: int, **kwargs) -> bool:
        # TODO: Реализуйте обновление элемента
        pass

    def delete(self, id_: int) -> bool:
        # TODO: Реализуйте удаление элемента
        pass