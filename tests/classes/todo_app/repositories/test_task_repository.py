import pytest
from datetime import datetime, timedelta
from src.classes.todo_app.models.task import Task
from src.classes.todo_app.repositories.task_repository import TaskRepository

@pytest.fixture
def task_repo(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.json")
    yield repo
    # Автоматическая очистка после теста

def test_add_and_get_simple_task(task_repo):
    task = Task(
        id=1,
        title="Test Task",
        description="Test Desc",
        category_id=1,
        status_id=1
    )
    added_task = task_repo.add(task)
    assert isinstance(added_task, Task)
    assert added_task.id == 1
    assert added_task.is_done is False
    assert task_repo.get(1).title == "Test Task"

def test_add_urgent_task_with_deadline(task_repo):
    deadline = datetime.now() + timedelta(days=7)
    task = Task(
        id=2,
        title="Urgent Task",
        description="Fix ASAP",
        category_id=1,
        status_id=1,
        deadline=deadline
    )
    added_task = task_repo.add(task)
    assert isinstance(added_task, Task)
    assert added_task.id == 2
    assert added_task.deadline == deadline

def test_add_recurring_task(task_repo):
    task = Task(
        id=3,
        title="Weekly Meeting",
        description="Team sync",
        category_id=1,
        status_id=1,
        repeat_every="week"
    )
    added_task = task_repo.add(task)
    assert isinstance(added_task, Task)
    assert added_task.id == 3
    assert added_task.repeat_every == "week"

def test_update_task_status(task_repo):
    task = Task(
        id=4,
        title="Task to Update",
        description="Test",
        category_id=1,
        status_id=1
    )
    task_repo.add(task)
    assert task_repo.update(4, status_id=2)
    updated_task = task_repo.get(4)
    assert updated_task.status_id == 2

def test_delete_task(task_repo):
    task = Task(
        id=5,
        title="Task to Delete",
        description="Test",
        category_id=1,
        status_id=1
    )
    task_repo.add(task)
    assert task_repo.delete(5)
    assert task_repo.get(5) is None

def test_add_task_with_existing_id(task_repo):
    # Добавляем первую задачу
    task1 = Task(
        id=6,
        title="First Task",
        description="Test",
        category_id=1,
        status_id=1
    )
    task_repo.add(task1)
    
    # Пытаемся добавить задачу с таким же ID
    task2 = Task(
        id=6,
        title="Duplicate Task",
        description="Should fail",
        category_id=1,
        status_id=1
    )
    with pytest.raises(ValueError, match="Item with id 6 already exists"):
        task_repo.add(task2)