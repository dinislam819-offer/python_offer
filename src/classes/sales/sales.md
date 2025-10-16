# Обработка данных о продажах из JSON файлов

## Задание

Даны 3 файла JSON с фиксированными именами:
- `products.json` - информация о товарах
- `customers.json` - информация о покупателях
- `orders.json` - информация о заказах

Все объекты связаны по id (числовое поле).

## Требования к реализации

1. **Модели данных** (уже реализованы):
   - `Product` - модель товара с полями `id`, `name`, `price`
   - `Customer` - модель покупателя с полями `id`, `name`
   - `Order` - модель заказа с полями `id`, `customer_id`, `product_ids`

2. **Класс анализатора** должен называться `SalesAnalyzer`
3. **Методы класса**:
   - `__init__(self, products, customers, orders)` - инициализация с тремя списками объектов
   - `analyze(self) -> Dict` - основной метод анализа, возвращающий словарь с результатами:
     ```python
     {
         "popular_product": {"name": str, "sales_count": int},
         "average_check": float,
         "total_revenue": float
     }
     ```
4. **Дополнительные функции**:
   - `load_data(products_path, customers_path, orders_path) -> SalesAnalyzer` - загрузка данных из JSON файлов
   - `print_results(analysis: Dict)` - вывод результатов в консоль

## Входные данные

Пример содержимого файлов:

`products.json`:
```json
[
    {"id": 1, "name": "Product A", "price": 100},
    {"id": 2, "name": "Product B", "price": 200}
]
```
`customers.json`:
```json
[
    {"id": 1, "name": "Customer X"},
    {"id": 2, "name": "Customer Y"}
]
```
`orders.json`:
```json
[
    {"id": 1, "customer_id": 1, "product_ids": [1, 2]},
    {"id": 2, "customer_id": 2, "product_ids": [1]}
]
```
Пример вывода:
Самый популярный товар: Product A (2 продажи)
Средний чек покупателя: 250.0
Общая выручка: 500

## Как запустить тесты

После выполнения задания проверьте своё решение с помощью автотестов:

```sh
python -m pytest tests/classes/test_sales.py
```

Убедитесь, что виртуальное окружение активировано перед запуском тестов.
