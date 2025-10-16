from typing import List
from src.classes.todo_app.models.task import Task
from src.classes.todo_app.repositories.base_repository import BaseRepository

class TaskRepository(BaseRepository[Task]):
    def __init__(self, file_path: str):
        # TODO: Реализуйте инициализацию
        pass

    def get_by_category(self, category_id: int) -> List[Task]:
        """Get all tasks belonging to a specific category."""
        # TODO: Реализуйте получение задач по категории
        pass

    def get_by_status(self, status_id: int) -> List[Task]:
        """Get all tasks with a specific status."""
        # TODO: Реализуйте получение задач по статусу
        pass