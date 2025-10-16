import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from src.classes.todo_app.models.task import Task
from src.classes.todo_app.models.category import Category
from src.classes.todo_app.models.status import Status
from src.classes.todo_app.repositories.task_repository import TaskRepository
from src.classes.todo_app.repositories.category_repository import CategoryRepository
from src.classes.todo_app.repositories.status_repository import StatusRepository


class TodoApp:
    """
    Основной класс приложения для управления задачами.
    Предоставляет доступ к репозиториям и методы с бизнес-логикой.
    """
    
    def __init__(self, data_dir: str):
        """
        Инициализация приложения.
        
        Args:
            data_dir: Путь к директории с JSON файлами данных
        """
        # TODO: Реализуйте инициализацию
        # Создайте репозитории и сохраните их как атрибуты:
        # self.task_repo = TaskRepository(...)
        # self.category_repo = CategoryRepository(...)
        # self.status_repo = StatusRepository(...)
        pass
    
    def add_task(self, title: str, category_id: int, status_id: int, **kwargs) -> Task:
        """
        Добавить новую задачу с проверкой существования категории и статуса.
        
        Args:
            title: Заголовок задачи
            category_id: ID категории
            status_id: ID статуса
            **kwargs: Дополнительные параметры (description, deadline, repeat_every)
            
        Returns:
            Созданная задача
            
        Raises:
            ValueError: Если категория или статус не существуют
        """
        # TODO: Реализуйте добавление задачи
        # 1. Проверьте существование категории и статуса
        # 2. Получите следующий ID для задачи
        # 3. Создайте задачу и добавьте её через репозиторий
        pass
    
    def mark_task_done(self, task_id: int) -> bool:
        """
        Отметить задачу как выполненную.
        
        Args:
            task_id: ID задачи
            
        Returns:
            True, если задача обновлена, False если не найдена
        """
        # TODO: Реализуйте отметку задачи как выполненной
        pass
    
    def get_overdue_tasks(self) -> List[Task]:
        """
        Получить просроченные задачи.
        
        Returns:
            Список задач с истекшим дедлайном
        """
        # TODO: Реализуйте получение просроченных задач
        # Получите все задачи и отфильтруйте те, у которых:
        # - есть дедлайн
        # - дедлайн истек (меньше текущего времени)
        # - задача не выполнена
        pass


def load_sample_data(app: TodoApp) -> None:
    """
    Загрузить примерные данные в приложение.
    
    Args:
        app: Экземпляр TodoApp
    """
    # TODO: Реализуйте загрузку примерных данных
    # Добавьте категории, статусы и примерные задачи
    pass


def print_task(task: Task) -> None:
    """
    Вывести информацию о задаче.
    
    Args:
        task: Задача для вывода
    """
    # TODO: Реализуйте вывод информации о задаче
    pass


def print_tasks(tasks: List[Task]) -> None:
    """
    Вывести список задач.
    
    Args:
        tasks: Список задач для вывода
    """
    # TODO: Реализуйте вывод списка задач
    pass


if __name__ == "__main__":
    # Пример использования
    app = TodoApp("src/classes/todo_app/data")
    
    # Загружаем примерные данные
    load_sample_data(app)
    
    print("=== Все задачи ===")
    # Теперь работаем с репозиторием напрямую
    all_tasks = app.task_repo.get_all()
    print_tasks(all_tasks)
    
    print("\n=== Задачи по категории 'Учеба' ===")
    study_tasks = app.task_repo.get_by_category(3)
    print_tasks(study_tasks)
    
    print("\n=== Просроченные задачи ===")
    overdue_tasks = app.get_overdue_tasks()
    print_tasks(overdue_tasks)
    
    # Добавляем новую задачу через высокоуровневый API
    print("\n=== Добавляем новую задачу ===")
    new_task = app.add_task(
        title="Новая задача",
        category_id=1,
        status_id=1,
        description="Это новая задача для демонстрации"
    )
    print_task(new_task) 