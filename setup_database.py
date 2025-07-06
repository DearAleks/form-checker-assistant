#!/usr/bin/env python3
"""
Скрипт для настройки векторной базы данных
Запускайте этот скрипт после добавления новых референсных документов
"""

import os
import sys
from dotenv import load_dotenv
from vector_db import VectorDatabase
from document_processor import DocumentProcessor

def setup_database():
    """Настройка базы данных с референсными документами"""
    
    # Загружаем переменные окружения
    load_dotenv()
    
    print("🔧 Настройка векторной базы данных...")
    
    # Инициализируем компоненты
    try:
        vector_db = VectorDatabase()
        doc_processor = DocumentProcessor()
        print("✅ Компоненты инициализированы")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False
    
    # Проверяем папку с референсными документами
    reference_docs_folder = os.getenv('REFERENCE_DOCS_FOLDER', './reference_docs')
    
    if not os.path.exists(reference_docs_folder):
        print(f"❌ Папка {reference_docs_folder} не найдена!")
        print(f"   Создайте папку и поместите туда PDF или TXT файлы с референсными документами")
        return False
    
    # Получаем список файлов
    files = [f for f in os.listdir(reference_docs_folder) 
             if f.endswith(('.pdf', '.txt'))]
    
    if not files:
        print(f"❌ В папке {reference_docs_folder} не найдены PDF или TXT файлы!")
        return False
    
    print(f"📚 Найдено файлов: {len(files)}")
    for file in files:
        print(f"   - {file}")
    
    # Обрабатываем документы
    print("\n🔄 Обработка документов...")
    try:
        documents = doc_processor.process_reference_documents(reference_docs_folder)
        
        if not documents:
            print("❌ Не удалось обработать документы!")
            return False
        
        print(f"✅ Обработано {len(documents)} частей документов")
        
    except Exception as e:
        print(f"❌ Ошибка обработки документов: {e}")
        return False
    
    # Очищаем старую базу данных
    print("\n🗑️  Очистка старой базы данных...")
    vector_db.clear_database()
    
    # Добавляем документы в базу данных
    print("\n💾 Добавление документов в базу данных...")
    success_count = 0
    
    for i, doc in enumerate(documents):
        try:
            if vector_db.add_document(doc['text'], doc):
                success_count += 1
            
            # Показываем прогресс
            if (i + 1) % 10 == 0:
                print(f"   Обработано: {i + 1}/{len(documents)}")
                
        except Exception as e:
            print(f"   ⚠️  Ошибка добавления документа {i}: {e}")
    
    # Сохраняем базу данных
    print("\n💾 Сохранение базы данных...")
    if vector_db.save_database():
        print("✅ База данных сохранена")
    else:
        print("❌ Ошибка сохранения базы данных")
        return False
    
    # Показываем статистику
    stats = vector_db.get_stats()
    print(f"\n📊 Статистика базы данных:")
    print(f"   Всего документов: {stats['total_documents']}")
    print(f"   Размерность эмбеддингов: {stats['embedding_dimension']}")
    print(f"   Модель: {stats['model_name']}")
    print(f"   Успешно добавлено: {success_count}")
    
    print("\n🎉 Настройка завершена успешно!")
    return True

def test_search():
    """Тестирование поиска"""
    print("\n🔍 Тестирование поиска...")
    
    vector_db = VectorDatabase()
    
    test_queries = [
        "õppekava",
        "mikrokvalifikatsioon",
        "õpiväljundid",
        "hindamiskriteeriumid"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Запрос: '{query}'")
        results = vector_db.search_similar(query, n_results=2)
        
        if results:
            for i, result in enumerate(results):
                print(f"   {i+1}. Сходство: {result['similarity']:.3f}")
                print(f"      Текст: {result['text'][:100]}...")
        else:
            print("   Результаты не найдены")

if __name__ == "__main__":
    print("🚀 Настройка векторной базы данных для Form Checker")
    print("="*60)
    
    if setup_database():
        # Опционально запускаем тест поиска
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            test_search()
    else:
        print("\n❌ Настройка не завершена из-за ошибок")
        sys.exit(1)