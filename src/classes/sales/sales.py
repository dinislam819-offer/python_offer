from typing import List, Dict
from pydantic import BaseModel
import json
from collections import defaultdict

# Модели Pydantic
class Product(BaseModel):
    id: int
    name: str
    price: float

class Customer(BaseModel):
    id: int
    name: str

class Order(BaseModel):
    id: int
    customer_id: int
    product_ids: List[int]

class SalesAnalyzer:
    def __init__(self, products: List[Product], customers: List[Customer], orders: List[Order]):
        # TODO: Реализуйте инициализацию
        pass

    def analyze(self) -> Dict:
        # TODO: Реализуйте метод анализа
        pass

def load_data(products_path: str, customers_path: str, orders_path: str) -> SalesAnalyzer:
    # TODO: Реализуйте загрузку данных
    pass

def print_results(analysis: Dict):
    # TODO: Реализуйте вывод результатов
    pass

# Пример использования
if __name__ == "__main__":
    analyzer = load_data("products.json", "customers.json", "orders.json")
    analysis = analyzer.analyze()
    print_results(analysis)