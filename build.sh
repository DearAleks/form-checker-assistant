#!/bin/bash
# Установка зависимостей
pip install -r requirements.txt

# Создание необходимых директорий
mkdir -p uploads
mkdir -p reference_docs

# Настройка базы данных (если есть референсные документы)
if [ -d "reference_docs" ] && [ "$(ls -A reference_docs)" ]; then
    echo "Настройка векторной базы данных..."
    python setup_database.py
else
    echo "Папка reference_docs пуста. Пропускаем настройку базы данных."
fi