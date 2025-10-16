import pytest
from src.classes.sales.sales import load_data
import json

def create_test_files(products, customers, orders, tmp_path):
    products_path = tmp_path / "products.json"
    customers_path = tmp_path / "customers.json"
    orders_path = tmp_path / "orders.json"
    
    with open(products_path, 'w') as f:
        json.dump(products, f)
    with open(customers_path, 'w') as f:
        json.dump(customers, f)
    with open(orders_path, 'w') as f:
        json.dump(orders, f)
    
    return products_path, customers_path, orders_path

def test_basic_case(tmp_path):
    products = [{"id": 1, "name": "A", "price": 100}, {"id": 2, "name": "B", "price": 200}]
    customers = [{"id": 1, "name": "X"}, {"id": 2, "name": "Y"}]
    orders = [
        {"id": 1, "customer_id": 1, "product_ids": [1, 2]},
        {"id": 2, "customer_id": 2, "product_ids": [1]}
    ]
    
    paths = create_test_files(products, customers, orders, tmp_path)
    analyzer = load_data(*paths)
    result = analyzer.analyze()
    
    assert result["popular_product"]["name"] == "A"
    assert result["popular_product"]["sales_count"] == 2
    assert result["average_check"] == 200.0
    assert result["total_revenue"] == 400.0

def test_empty_orders(tmp_path):
    products = [{"id": 1, "name": "A", "price": 100}]
    customers = [{"id": 1, "name": "X"}]
    orders = []
    
    paths = create_test_files(products, customers, orders, tmp_path)
    analyzer = load_data(*paths)
    result = analyzer.analyze()
    
    assert result["popular_product"]["name"] is None
    assert result["popular_product"]["sales_count"] == 0
    assert result["average_check"] == 0
    assert result["total_revenue"] == 0.0

def test_single_product_multiple_orders(tmp_path):
    products = [{"id": 1, "name": "A", "price": 100}]
    customers = [{"id": 1, "name": "X"}]
    orders = [
        {"id": 1, "customer_id": 1, "product_ids": [1]},
        {"id": 2, "customer_id": 1, "product_ids": [1]}
    ]
    
    paths = create_test_files(products, customers, orders, tmp_path)
    analyzer = load_data(*paths)
    result = analyzer.analyze()
    
    assert result["popular_product"]["name"] == "A"
    assert result["popular_product"]["sales_count"] == 2
    assert result["average_check"] == 100.0
    assert result["total_revenue"] == 200.0

def test_tie_in_popularity(tmp_path):
    products = [
        {"id": 1, "name": "A", "price": 100},
        {"id": 2, "name": "B", "price": 200}
    ]
    customers = [{"id": 1, "name": "X"}]
    orders = [
        {"id": 1, "customer_id": 1, "product_ids": [1, 2]}
    ]
    
    paths = create_test_files(products, customers, orders, tmp_path)
    analyzer = load_data(*paths)
    result = analyzer.analyze()
    
    # Должен вернуть первый товар с максимальными продажами (в данном случае любой из двух)
    assert result["popular_product"]["sales_count"] == 1
    assert result["average_check"] == 300.0
    assert result["total_revenue"] == 300.0

def test_multiple_products_in_order(tmp_path):
    products = [
        {"id": 1, "name": "A", "price": 100},
        {"id": 2, "name": "B", "price": 200},
        {"id": 3, "name": "C", "price": 300}
    ]
    customers = [{"id": 1, "name": "X"}]
    orders = [
        {"id": 1, "customer_id": 1, "product_ids": [1, 2, 3]}
    ]
    
    paths = create_test_files(products, customers, orders, tmp_path)
    analyzer = load_data(*paths)
    result = analyzer.analyze()
    
    assert result["popular_product"]["sales_count"] == 1
    assert result["average_check"] == 600.0
    assert result["total_revenue"] == 600.0

def test_non_existent_product_reference(tmp_path):
    products = [{"id": 1, "name": "A", "price": 100}]
    customers = [{"id": 1, "name": "X"}]
    orders = [
        {"id": 1, "customer_id": 1, "product_ids": [1, 999]}
    ]
    
    paths = create_test_files(products, customers, orders, tmp_path)
    analyzer = load_data(*paths)
    
    with pytest.raises(KeyError):
        analyzer.analyze()