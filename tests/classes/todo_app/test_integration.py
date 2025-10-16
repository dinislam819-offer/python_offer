import pytest
import os
from pathlib import Path
from datetime import datetime, timedelta
from src.classes.todo_app.models.task import Task
from src.classes.todo_app.models.category import Category
from src.classes.todo_app.models.status import Status
from src.classes.todo_app.repositories.task_repository import TaskRepository
from src.classes.todo_app.repositories.category_repository import CategoryRepository
from src.classes.todo_app.repositories.status_repository import StatusRepository

class System:
    def __init__(self, storage_dir: str = "data"):
        self._storage_dir = Path(storage_dir)
        os.makedirs(self._storage_dir, exist_ok=True)
        
        self.tasks = TaskRepository(self._storage_dir / "tasks.json")
        self.categories = CategoryRepository(self._storage_dir / "categories.json")
        self.statuses = StatusRepository(self._storage_dir / "statuses.json")
        self._init_default_data()

    def _init_default_data(self):
        """Инициализирует обязательные данные"""
        if not self.categories.get_by_name("Работа"):
            self.categories.add(Category(id=1, name="Работа"))
        if not self.categories.get_by_name("Личное"):
            self.categories.add(Category(id=2, name="Личное"))

    def clear_all_data(self):
        """Очищает все данные (для тестов)"""
        self.tasks._data.clear()
        self.categories._data.clear()
        self.statuses._data.clear()
        self._init_default_data()

@pytest.fixture
def populated_system(tmp_path):
    """Фикстура с предзаполненной системой, использующая только репозитории"""
    system = System(storage_dir=str(tmp_path / "data"))
    
    # Очищаем дефолтные данные для чистоты тестов
    system.clear_all_data()
    
    # Добавляем тестовые категории через репозиторий
    system.categories.add(Category(id=10, name="Разработка"))
    system.categories.add(Category(id=11, name="Тестирование"))
    system.categories.add(Category(id=12, name="Документация"))
    
    # Добавляем задачи разных типов через репозиторий
    system.tasks.add(Task(
        id=1,
        title="Рефакторинг модуля А",
        description="Улучшить структуру кода",
        category_id=10,
        status_id=1
    ))
    system.tasks.add(Task(
        id=2,
        title="Написать unit-тесты",
        category_id=11,
        status_id=2,
        is_done=True
    ))
    system.tasks.add(Task(
        id=3,
        title="Исправить критический баг",
        category_id=10,
        status_id=2,
        deadline=datetime.now() + timedelta(hours=12)
    ))
    system.tasks.add(Task(
        id=4,
        title="Еженедельный митинг",
        category_id=12,
        status_id=1,
        repeat_every="week"
    ))
    
    return system

def test_bulk_status_update(populated_system):
    """Массовое обновление статусов задач по категории"""
    # Получаем ID категории "Разработка" через поиск по имени
    dev_category = next(
        c for c in populated_system.categories.get_all() 
        if c.name == "Разработка"
    )
    
    # Получаем все задачи категории через репозиторий
    dev_tasks = [
        t for t in populated_system.tasks.get_all() 
        if t.category_id == dev_category.id
    ]
    assert len(dev_tasks) >= 2
    
    # Обновляем статусы
    for task in dev_tasks:
        populated_system.tasks.update(task.id, status_id=2)
    
    # Проверяем через репозиторий
    updated_tasks = [
        t for t in populated_system.tasks.get_all() 
        if t.category_id == dev_category.id and t.status_id == 2
    ]
    assert len(updated_tasks) == len(dev_tasks)