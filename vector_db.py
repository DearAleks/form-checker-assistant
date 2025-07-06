import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VectorDatabase:
    def __init__(self):
        """Инициализация векторной базы данных с локальной моделью"""
        print("Загрузка локальной модели для эмбеддингов...")
        # Используем компактную, но эффективную модель
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Модель загружена успешно!")
        
        self.embeddings = []
        self.documents = []
        self.metadata = []
        
        # Файл для сохранения данных
        self.db_file = 'vector_database.pkl'
        
        # Загружаем существующие данные, если есть
        self.load_database()
    
    def create_embeddings(self, texts):
        """Создает эмбеддинги для списка текстов"""
        try:
            if not texts:
                return []
            
            print(f"Создание эмбеддингов для {len(texts)} текстов...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Ошибка при создании эмбеддингов: {e}")
            return []
    
    def add_document(self, text, metadata=None):
        """Добавляет документ в векторную базу данных"""
        try:
            # Создаем эмбеддинг для текста
            embedding = self.create_embeddings([text])
            
            if not embedding:
                print("Не удалось создать эмбеддинг для документа")
                return False
            
            # Добавляем в базу данных
            self.embeddings.append(embedding[0])
            self.documents.append(text)
            self.metadata.append(metadata or {})
            
            print(f"Документ добавлен. Всего документов: {len(self.documents)}")
            return True
            
        except Exception as e:
            print(f"Ошибка при добавлении документа: {e}")
            return False
    
    def search_similar(self, query, n_results=3):
        """Поиск похожих документов"""
        try:
            if not self.embeddings:
                print("База данных пуста")
                return []
            
            # Создаем эмбеддинг для запроса
            query_embeddings = self.create_embeddings([query])
            
            if not query_embeddings:
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
    
    def save_database(self):
        """Сохраняет базу данных в файл"""
        try:
            data = {
                'embeddings': self.embeddings,
                'documents': self.documents,
                'metadata': self.metadata
            }
            
            with open(self.db_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"База данных сохранена в {self.db_file}")
            return True
            
        except Exception as e:
            print(f"Ошибка при сохранении базы данных: {e}")
            return False
    
    def load_database(self):
        """Загружает базу данных из файла"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.embeddings = data.get('embeddings', [])
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                
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
        
        # Удаляем файл базы данных
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
        
        print("База данных очищена")
    
    def get_stats(self):
        """Возвращает статистику базы данных"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0,
            'model_name': 'all-MiniLM-L6-v2'
        }