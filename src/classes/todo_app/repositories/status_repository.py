from src.classes.todo_app.models.status import Status
from src.classes.todo_app.repositories.base_repository import BaseRepository

class StatusRepository(BaseRepository[Status]):
    def __init__(self, file_path: str):
        # TODO: Реализуйте инициализацию
        pass

    def _init_default_statuses(self) -> None:
        # TODO: Реализуйте инициализацию статусов по умолчанию
        pass

    def is_valid_status(self, status_id: int) -> bool:
        # TODO: Реализуйте проверку валидности статуса
        pass