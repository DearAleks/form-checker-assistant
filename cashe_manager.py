#!/usr/bin/env python3
"""
Менеджер кеша эмбеддингов
Позволяет управлять кешем эмбеддингов
"""

import os
import sys
import argparse
import pickle
import time
from dotenv import load_dotenv
from vector_db import VectorDatabase

def show_cache_stats():
    """Показывает статистику кеша"""
    print("📊 Статистика кеша эмбеддингов")
    print("=" * 40)
    
    try:
        vector_db = VectorDatabase()
        stats = vector_db.get_cache_stats()
        
        print(f"Кеширование включено: {'✅' if stats['cache_enabled'] else '❌'}")
        
        if stats['cache_enabled']:
            print(f"Файлов в кеше: {stats['cache_files']}")
            print(f"Размер кеша: {stats['cache_size_mb']:.2f} MB")
            print(f"Время жизни: {stats['cache_ttl_hours']:.1f} часов")
            
            # Показываем детали по провайдерам
            cache_dir = os.getenv('CACHE_DIR', './embeddings_cache')
            if os.path.exists(cache_dir):
                print(f"\nПапка кеша: {cache_dir}")
                
                # Анализируем кеш файлы
                provider_stats = {}
                for filename in os.listdir(cache_dir):
                    if filename.endswith('.pkl'):
                        filepath = os.path.join(cache_dir, filename)
                        try:
                            with open(filepath, 'rb') as f:
                                data = pickle.load(f)
                                provider = data.get('provider', 'unknown')
                                model = data.get('model', 'unknown')
                                key = f"{provider}:{model}"
                                
                                if key not in provider_stats:
                                    provider_stats[key] = 0
                                provider_stats[key] += 1
                        except:
                            continue
                
                if provider_stats:
                    print("\nРаспределение по провайдерам:")
                    for key, count in provider_stats.items():
                        print(f"  {key}: {count} файлов")
        
    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")

def clear_cache():
    """Очищает кеш"""
    print("🧹 Очистка кеша эмбеддингов...")
    
    try:
        vector_db = VectorDatabase()
        vector_db.clear_cache()
        print("✅ Кеш успешно очищен")
    except Exception as e:
        print(f"❌ Ошибка очистки кеша: {e}")

def cleanup_old_cache():
    """Удаляет устаревшие файлы кеша"""
    print("🗑️  Удаление устаревших файлов кеша...")
    
    try:
        cache_dir = os.getenv('CACHE_DIR', './embeddings_cache')
        cache_ttl = int(os.getenv('CACHE_TTL', '86400'))
        
        if not os.path.exists(cache_dir):
            print("Папка кеша не найдена")
            return
        
        current_time = time.time()
        removed_count = 0
        
        for filename in os.listdir(cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(cache_dir, filename)
                file_age = current_time - os.path.getmtime(filepath)
                
                if file_age > cache_ttl:
                    os.remove(filepath)
                    removed_count += 1
        
        print(f"✅ Удалено {removed_count} устаревших файлов")
        
    except Exception as e:
        print(f"❌ Ошибка удаления устаревших файлов: {e}")

def migrate_cache(old_provider, new_provider):
    """Помогает мигрировать кеш при смене провайдера"""
    print(f"🔄 Миграция кеша с {old_provider} на {new_provider}")
    
    try:
        cache_dir = os.getenv('CACHE_DIR', './embeddings_cache')
        
        if not os.path.exists(cache_dir):
            print("Папка кеша не найдена")
            return
        
        # Создаем папку для старого кеша
        old_cache_dir = f"{cache_dir}_backup_{old_provider}_{int(time.time())}"
        os.makedirs(old_cache_dir, exist_ok=True)
        
        moved_count = 0
        
        for filename in os.listdir(cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(cache_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        if data.get('provider') == old_provider:
                            # Перемещаем файл в резервную папку
                            old_filepath = os.path.join(old_cache_dir, filename)
                            os.rename(filepath, old_filepath)
                            moved_count += 1
                except:
                    continue
        
        print(f"✅ Перемещено {moved_count} файлов в {old_cache_dir}")
        print(f"   Новые эмбеддинги будут созданы с провайдером {new_provider}")
        
    except Exception as e:
        print(f"❌ Ошибка миграции кеша: {e}")

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Менеджер кеша эмбеддингов")
    parser.add_argument('action', choices=['stats', 'clear', 'cleanup', 'migrate'], 
                       help='Действие для выполнения')
    parser.add_argument('--old-provider', help='Старый провайдер для миграции')
    parser.add_argument('--new-provider', help='Новый провайдер для миграции')
    
    args = parser.parse_args()
    
    if args.action == 'stats':
        show_cache_stats()
    elif args.action == 'clear':
        clear_cache()
    elif args.action == 'cleanup':
        cleanup_old_cache()
    elif args.action == 'migrate':
        if not args.old_provider or not args.new_provider:
            print("❌ Для миграции нужно указать --old-provider и --new-provider")
            sys.exit(1)
        migrate_cache(args.old_provider, args.new_provider)

if __name__ == "__main__":
    main()