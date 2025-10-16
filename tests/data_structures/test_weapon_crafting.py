from src.data_structures.weapon_crafting.weapon_crafting import find_most_expensive_weapons

def test_basic_case():
    inventory = {"wood": 5, "metal": 10, "rope": 3}
    blueprints = {
        "knife": {"materials": {"wood": 1, "metal": 2}, "price": 10},
        "sword": {"materials": {"wood": 2, "metal": 5}, "price": 20},
        "axe": {"materials": {"wood": 3, "metal": 4}, "price": 20},
        "hammer": {"materials": {"metal": 8, "rope": 1}, "price": 25},
    }
    assert find_most_expensive_weapons(inventory, blueprints) == ["hammer"]


def test_multiple_max_priced():
    inventory = {"wood": 10, "metal": 10}
    blueprints = {
        "sword": {"materials": {"wood": 2, "metal": 5}, "price": 20},
        "axe": {"materials": {"wood": 3, "metal": 4}, "price": 20},
        "shield": {"materials": {"wood": 5, "metal": 5}, "price": 15},
    }
    result = find_most_expensive_weapons(inventory, blueprints)
    assert set(result) == {"sword", "axe"}
    assert len(result) == 2


def test_no_materials():
    inventory = {"wood": 0, "metal": 0}
    blueprints = {
        "knife": {"materials": {"wood": 1, "metal": 1}, "price": 10},
    }
    assert find_most_expensive_weapons(inventory, blueprints) == []


def test_not_enough_for_any():
    inventory = {"wood": 1, "metal": 1}
    blueprints = {
        "sword": {"materials": {"wood": 2, "metal": 2}, "price": 20},
        "knife": {"materials": {"wood": 1, "metal": 2}, "price": 10},
    }
    assert find_most_expensive_weapons(inventory, blueprints) == []


def test_exact_materials():
    inventory = {"wood": 2, "metal": 4}
    blueprints = {
        "item1": {"materials": {"wood": 2, "metal": 4}, "price": 30},
        "item2": {"materials": {"wood": 1, "metal": 1}, "price": 10},
    }
    assert find_most_expensive_weapons(inventory, blueprints) == ["item1"]


def test_missing_material():
    inventory = {"wood": 10}
    blueprints = {
        "wand": {"materials": {"wood": 2, "gem": 1}, "price": 50},
    }
    assert find_most_expensive_weapons(inventory, blueprints) == []


def test_empty_inventory_and_blueprints():
    assert find_most_expensive_weapons({}, {}) == []


def test_zero_quantity_material():
    inventory = {"wood": 5, "metal": 0}
    blueprints = {
        "knife": {"materials": {"wood": 1, "metal": 1}, "price": 10},
    }
    assert find_most_expensive_weapons(inventory, blueprints) == []