import pytest
from src.classes.todo_app.models.status import Status
from src.classes.todo_app.repositories.status_repository import StatusRepository

@pytest.fixture
def status_repo(tmp_path):
    repo = StatusRepository(tmp_path / "statuses.json")
    yield repo

def test_default_statuses_auto_created(status_repo):
    statuses = status_repo.get_all()
    assert len(statuses) == 3
    assert {s.name for s in statuses} == {"В ожидании", "В работе", "Завершено"}

def test_is_valid_status(status_repo):
    assert status_repo.is_valid_status(1) is True
    assert status_repo.is_valid_status(999) is False

def test_add_custom_status(status_repo):
    status_repo.add(Status(id=4, name="Отменено"))
    assert status_repo.get(4).name == "Отменено"