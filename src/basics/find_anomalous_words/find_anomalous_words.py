import re

def find_anomalous_words(text: str) -> list[str]:
    """
    Находит слова, длина которых отличается от средней длины слов в тексте более чем на 2 символа.

    

    :param text: Входная строка.
    :return: Список аномальных слов.
    """
    # TODO: Реализуйте функцию
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return []
    total_length = 0
    for word in words:
        total_length += len(word)
    avg_length = total_length / len(words)

    result = []
    for word in words:
        if abs(len(word) - avg_length) >= 2:
            result.append(word)

    return result