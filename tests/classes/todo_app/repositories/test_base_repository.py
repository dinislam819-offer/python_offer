import pytest
import json
import os
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

# Assuming the BaseRepository is in repository.py
from src.classes.todo_app.repositories.base_repository import BaseRepository, DateTimeEncoder

# Define a test model - renamed to avoid pytest collection warning
class SampleTestModel(BaseModel):
    id: int  # Теперь id обязательный, без Optional
    name: str
    value: int
    created_at: Optional[datetime] = None

@pytest.fixture
def test_file(tmp_path):
    test_file = tmp_path / "test_db.json"
    yield str(test_file)
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

@pytest.fixture
def repo(test_file):
    return BaseRepository(test_file, SampleTestModel)

def test_datetime_encoder():
    encoder = DateTimeEncoder()
    test_date = datetime(2023, 1, 1, 12, 0, 0)
    encoded = encoder.encode({"date": test_date})
    assert '"2023-01-01T12:00:00"' in encoded

def test_repository_initialization(repo, test_file):
    assert repo._file_path == test_file
    assert repo._model == SampleTestModel
    assert repo._data == {}
    assert not os.path.exists(test_file)

def test_load_existing_data(tmp_path, test_file):
    # Create a test file with data
    data = {
        "1": {"id": 1, "name": "test", "value": 10, "created_at": "2023-01-01T00:00:00"},
        "2": {"id": 2, "name": "test2", "value": 20, "created_at": "2023-01-02T00:00:00"}
    }
    with open(test_file, "w") as f:
        json.dump(data, f)
    
    repo = BaseRepository(test_file, SampleTestModel)
    assert len(repo._data) == 2
    assert 1 in repo._data
    assert 2 in repo._data
    assert isinstance(repo._data[1].created_at, datetime)

def test_add_item(repo):
    item = SampleTestModel(id=1, name="test", value=10)
    result = repo.add(item)
    
    assert result.id == 1
    assert len(repo._data) == 1
    assert 1 in repo._data
    assert repo._data[1].name == "test"
    assert os.path.exists(repo._file_path)

def test_add_item_with_existing_id(repo):
    item1 = SampleTestModel(id=1, name="test", value=10)
    repo.add(item1)
    
    item2 = SampleTestModel(id=1, name="test2", value=20)
    with pytest.raises(ValueError, match="Item with id 1 already exists"):
        repo.add(item2)

def test_get_item(repo):
    item = SampleTestModel(id=1, name="test", value=10)
    repo.add(item)
    
    result = repo.get(1)
    assert result is not None
    assert result.name == "test"
    assert result.value == 10
    
    assert repo.get(999) is None

def test_get_all_items(repo):
    item1 = SampleTestModel(id=1, name="test1", value=10)
    item2 = SampleTestModel(id=2, name="test2", value=20)
    repo.add(item1)
    repo.add(item2)
    
    results = repo.get_all()
    assert len(results) == 2
    assert results[0].name == "test1"
    assert results[1].name == "test2"

def test_update_item(repo):
    item = SampleTestModel(id=1, name="test", value=10)
    repo.add(item)
    
    # Successful update
    assert repo.update(1, name="updated", value=100)
    updated = repo.get(1)
    assert updated.name == "updated"
    assert updated.value == 100
    
    # Failed update (non-existent ID)
    assert not repo.update(999, name="nonexistent")

def test_delete_item(repo):
    item = SampleTestModel(id=1, name="test", value=10)
    repo.add(item)
    
    # Successful delete
    assert repo.delete(1)
    assert len(repo._data) == 0
    assert repo.get(1) is None
    
    # Failed delete (non-existent ID)
    assert not repo.delete(999)

def test_save_load_cycle(repo):
    item1 = SampleTestModel(id=1, name="test1", value=10, created_at=datetime.now())
    item2 = SampleTestModel(id=2, name="test2", value=20, created_at=datetime.now())
    repo.add(item1)
    repo.add(item2)
    
    # Create a new repo instance to test loading
    new_repo = BaseRepository(repo._file_path, SampleTestModel)
    assert len(new_repo._data) == 2
    assert new_repo.get(1).name == "test1"
    assert new_repo.get(2).name == "test2"
    assert isinstance(new_repo.get(1).created_at, datetime)

def test_model_with_datetime(repo):
    test_date = datetime(2023, 1, 1, 12, 0, 0)
    item = SampleTestModel(id=1, name="test", value=10, created_at=test_date)
    repo.add(item)
    
    # Check if datetime was properly saved and loaded
    new_repo = BaseRepository(repo._file_path, SampleTestModel)
    loaded_item = new_repo.get(1)
    assert loaded_item.created_at == test_date