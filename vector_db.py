import os
import pickle
import numpy as np
import hashlib
import time
from typing import List, Dict, Optional
import requests
import openai
from sklearn.metrics.pairwise import cosine_similarity

class VectorDatabase:
    def __init__(self):
        """Инициализация векторной базы данных с внешними API"""
        print("Инициализация векторной базы данных с внешними API...")
        
        # Настройка API
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        # Выбираем провайдера эмбеддингов
        self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai')  # openai, huggingface
        
        # Настройки для разных провайдеров
        self.embedding_configs = {
            'openai': {
                'model': 'text-embedding-3-small',  # Более дешевая модель
                'dimensions': 1536,
                'max_tokens': 8191
            },
            'huggingface': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'api_url': 'https://api-inference.huggingface.co/pipeline/feature-extraction/',
                'dimensions': 384
            }
        }
        
        self.embeddings = []
        self.documents = []
        self.metadata = []
        
        # Файлы для сохранения данных и кеша
        self.db_file = 'vector_database.pkl'
        self.cache_file = 'embedding_cache.pkl'
        
        # Настройки кеширования
        self.cache_ttl = int(os.getenv('CACHE_TTL', '86400'))  # 24 часа по умолчанию
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE', '1000'))  # Максимум записей в кеше
        
        # Загружаем кеш и данные
        self.embedding_cache = self.load_cache()
        self.load_database()
        
        print(f"Используется провайдер эмбеддингов: {self.embedding_provider}")
    
    def _get_cache_key(self, text: str) -> str:
        """Создает ключ для кеширования на основе хеша текста"""
        return hashlib.md5(f"{self.embedding_provider}:{text}".encode()).hexdigest()
    
    def load_cache(self) -> Dict:
        """Загружает кеш эмбеддингов"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                
                # Очищаем устаревшие записи
                current_time = time.time()
                valid_cache = {}
                
                for key, value in cache.items():
                    if current_time - value['timestamp'] < self.cache_ttl:
                        valid_cache[key] = value
                
                print(f"Загружен кеш с {len(valid_cache)} записями")
                return valid_cache
            else:
                print("Файл кеша не найден, создается новый")
                return {}
                
        except Exception as e:
            print(f"Ошибка при загрузке кеша: {e}")
            return {}
    
    def save_cache(self):
        """Сохраняет кеш эмбеддингов"""
        try:
            # Ограничиваем размер кеша
            if len(self.embedding_cache) > self.max_cache_size:
                # Удаляем самые старые записи
                sorted_items = sorted(
                    self.embedding_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                self.embedding_cache = dict(sorted_items[-self.max_cache_size:])
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            print(f"Кеш сохранен ({len(self.embedding_cache)} записей)")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении кеша: {e}")
            return False
    
    def get_embedding_from_cache(self, text: str) -> Optional[List[float]]:
        """Получает эмбеддинг из кеша"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.embedding_cache:
            cached_data = self.embedding_cache[cache_key]
            
            # Проверяем, не устарел ли кеш
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                print("Эмбеддинг найден в кеше")
                return cached_data['embedding']
            else:
                # Удаляем устаревший кеш
                del self.embedding_cache[cache_key]
        
        return None
    
    def save_embedding_to_cache(self, text: str, embedding: List[float]):
        """Сохраняет эмбеддинг в кеш"""
        cache_key = self._get_cache_key(text)
        self.embedding_cache[cache_key] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
    
    def create_embeddings_openai(self, texts: List[str]) -> List[List[float]]:
        """Создает эмбеддинги через OpenAI API"""
        try:
            if not self.openai_api_key:
                raise ValueError("OpenAI API ключ не установлен")
            
            config = self.embedding_configs['openai']
            
            # Обрабатываем тексты батчами (OpenAI поддерживает до 2048 текстов за раз)
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = openai.embeddings.create(
                    input=batch,
                    model=config['model']
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"Обработано {min(i + batch_size, len(texts))}/{len(texts)} текстов")
            
            return all_embeddings
            
        except Exception as e:
            print(f"Ошибка при создании эмбеддингов через OpenAI: {e}")
            return []
    
    def create_embeddings_huggingface(self, texts: List[str]) -> List[List[float]]:
        """Создает эмбеддинги через Hugging Face API"""
        try:
            if not self.huggingface_api_key:
                raise ValueError("Hugging Face API ключ не установлен")
            
            config = self.embedding_configs['huggingface']
            headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
            
            all_embeddings = []
            
            for text in texts:
                response = requests.post(
                    f"{config['api_url']}{config['model']}",
                    headers=headers,
                    json={"inputs": text}
                )
                
                if response.status_code == 200:
                    embedding = response.json()
                    all_embeddings.append(embedding)
                else:
                    print(f"Ошибка API Hugging Face: {response.status_code}")
                    return []
            
            return all_embeddings
            
        except Exception as e:
            print(f"Ошибка при создании эмбеддингов через Hugging Face: {e}")
            return []
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает эмбеддинги с учетом кеширования"""
        if not texts:
            return []
        
        # Проверяем кеш для каждого текста
        cached_embeddings = []
        texts_to_process = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.get_embedding_from_cache(text)
            if cached_embedding:
                cached_embeddings.append((i, cached_embedding))
            else:
                texts_to_process.append(text)
                cache_indices.append(i)
        
        # Создаем эмбеддинги для некешированных текстов
        new_embeddings = []
        if texts_to_process:
            print(f"Создание эмбеддингов для {len(texts_to_process)} новых текстов...")
            
            if self.embedding_provider == 'openai':
                new_embeddings = self.create_embeddings_openai(texts_to_process)
            elif self.embedding_provider == 'huggingface':
                new_embeddings = self.create_embeddings_huggingface(texts_to_process)
            else:
                raise ValueError(f"Неизвестный провайдер эмбеддингов: {self.embedding_provider}")
            
            # Сохраняем новые эмбеддинги в кеш
            for text, embedding in zip(texts_to_process, new_embeddings):
                self.save_embedding_to_cache(text, embedding)
        
        # Объединяем кешированные и новые эмбеддинги
        all_embeddings = [None] * len(texts)
        
        # Вставляем кешированные эмбеддинги
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        # Вставляем новые эмбеддинги
        for i, embedding in zip(cache_indices, new_embeddings):
            all_embeddings[i] = embedding
        
        # Сохраняем кеш
        if new_embeddings:
            self.save_cache()
        
        return all_embeddings
    
    def add_document(self, text: str, metadata: Dict = None) -> bool:
        """Добавляет документ в векторную базу данных"""
        try:
            # Создаем эмбеддинг для текста
            embeddings = self.create_embeddings([text])
            
            if not embeddings or not embeddings[0]:
                print("Не удалось создать эмбеддинг для документа")
                return False
            
            # Добавляем в базу данных
            self.embeddings.append(embeddings[0])
            self.documents.append(text)
            self.metadata.append(metadata or {})
            
            print(f"Документ добавлен. Всего документов: {len(self.documents)}")
            return True
            
        except Exception as e:
            print(f"Ошибка при добавлении документа: {e}")
            return False
    
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """Поиск похожих документов"""
        try:
            if not self.embeddings:
                print("База данных пуста")
                return []
            
            # Создаем эмбеддинг для запроса
            query_embeddings = self.create_embeddings([query])
            
            if not query_embeddings or not query_embeddings[0]:
                print("Не удалось создать эмбеддинг для запроса")
                return []
            
            query_embedding = np.array(query_embeddings[0]).reshape(1, -1)
            doc_embeddings = np.array(self.embeddings)
            
            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Находим топ-N наиболее похожих документов
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Минимальный порог сходства
                    results.append({
                        'text': self.documents[idx],
                        'similarity': float(similarities[idx]),
                        'metadata': self.metadata[idx]
                    })
            
            print(f"Найдено {len(results)} похожих документов")
            return results
            
        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            return []
    
    def save_database(self) -> bool:
        """Сохраняет базу данных в файл"""
        try:
            data = {
                'embeddings': self.embeddings,
                'documents': self.documents,
                'metadata': self.metadata,
                'provider': self.embedding_provider,
                'config': self.embedding_configs[self.embedding_provider]
            }
            
            with open(self.db_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"База данных сохранена в {self.db_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении базы данных: {e}")
            return False
    
    def load_database(self) -> bool:
        """Загружает базу данных из файла"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.embeddings = data.get('embeddings', [])
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                
                # Проверяем совместимость провайдера
                saved_provider = data.get('provider', 'unknown')
                if saved_provider != self.embedding_provider:
                    print(f"⚠️  Внимание: База данных создана с провайдером {saved_provider}, "
                          f"но текущий провайдер {self.embedding_provider}")
                    print("Рекомендуется пересоздать базу данных")
                
                print(f"Загружена база данных с {len(self.documents)} документами")
                return True
            else:
                print("Файл базы данных не найден, создается новая база")
                return False
                
        except Exception as e:
            print(f"Ошибка при загрузке базы данных: {e}")
            return False
    
    def clear_database(self):
        """Очищает базу данных"""
        self.embeddings = []
        self.documents = []
        self.metadata = []
        
        # Удаляем файлы базы данных и кеша
        for file_path in [self.db_file, self.cache_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self.embedding_cache = {}
        print("База данных и кеш очищены")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику базы данных"""
        config = self.embedding_configs[self.embedding_provider]
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': config.get('dimensions', 
                                            len(self.embeddings[0]) if self.embeddings else 0),
            'model_name': config.get('model', 'unknown'),
            'provider': self.embedding_provider,
            'cache_size': len(self.embedding_cache),
            'cache_ttl_hours': self.cache_ttl / 3600
        }
    
    def get_cache_stats(self) -> Dict:
        """Возвращает статистику кеша"""
        current_time = time.time()
        valid_entries = sum(1 for entry in self.embedding_cache.values() 
                          if current_time - entry['timestamp'] < self.cache_ttl)
        
        return {
            'total_entries': len(self.embedding_cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.embedding_cache) - valid_entries,
            'cache_size_mb': os.path.getsize(self.cache_file) / (1024 * 1024) if os.path.exists(self.cache_file) else 0
        }
    
    def cleanup_cache(self):
        """Очищает устаревшие записи кеша"""
        current_time = time.time()
        valid_cache = {}
        
        for key, value in self.embedding_cache.items():
            if current_time - value['timestamp'] < self.cache_ttl:
                valid_cache[key] = value
        
        removed_count = len(self.embedding_cache) - len(valid_cache)
        self.embedding_cache = valid_cache
        
        if removed_count > 0:
            self.save_cache()
            print(f"Удалено {removed_count} устаревших записей из кеша")
        
        return removed_count