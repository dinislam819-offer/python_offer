import pytest
from src.classes.todo_app.models.category import Category
from src.classes.todo_app.repositories.category_repository import CategoryRepository

@pytest.fixture
def category_repo(tmp_path):
    repo = CategoryRepository(tmp_path / "categories.json")
    yield repo

def test_add_and_get_category(category_repo):
    category = category_repo.add(Category(id=1, name="Work"))
    assert category_repo.get(1).name == "Work"

def test_get_by_name_case_insensitive(category_repo):
    category_repo.add(Category(id=1, name="Work"))
    assert category_repo.get_by_name("WORK").name == "Work"
    assert category_repo.get_by_name("unknown") is None

def test_update_category(category_repo):
    category_repo.add(Category(id=1, name="Old Name"))
    assert category_repo.update(1, name="New Name")
    assert category_repo.get(1).name == "New Name"

def test_delete_category(category_repo):
    category_repo.add(Category(id=1, name="To Delete"))
    assert category_repo.delete(1)
    assert category_repo.get(1) is None