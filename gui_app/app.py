"""
GUI приложение для RAG ассистента.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import threading

# Добавляем пути к модулям
sys.path.append(str(Path(__file__).parent.parent / 'assistant_api'))
sys.path.append(str(Path(__file__).parent.parent / 'assistant_giga'))

# Загрузка .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


class RAGAssistantGUI:
    """GUI для RAG ассистента."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Ассистент")
        self.root.geometry("900x700")
        
        self.pipeline = None
        self.mode = tk.StringVar(value="api")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Создание виджетов интерфейса."""
        
        # Верхняя панель - выбор режима
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Режим работы:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(top_frame, text="OpenAI API", variable=self.mode, 
                       value="api", command=self._on_mode_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(top_frame, text="GigaChat", variable=self.mode, 
                       value="giga", command=self._on_mode_change).pack(side=tk.LEFT, padx=5)
        
        self.init_button = ttk.Button(top_frame, text="Инициализировать", 
                                     command=self._initialize_pipeline)
        self.init_button.pack(side=tk.LEFT, padx=20)
        
        self.status_label = ttk.Label(top_frame, text="Не инициализировано", 
                                     foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Область чата
        chat_frame = ttk.LabelFrame(self.root, text="Диалог", padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, 
                                                      font=("Arial", 10), state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Настройка тегов для форматирования
        self.chat_display.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("assistant", foreground="green", font=("Arial", 10, "bold"))
        self.chat_display.tag_config("system", foreground="gray", font=("Arial", 9, "italic"))
        self.chat_display.tag_config("cache", foreground="orange")
        
        # Панель ввода
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Ваш вопрос:").pack(anchor=tk.W)
        
        self.query_entry = ttk.Entry(input_frame, font=("Arial", 10))
        self.query_entry.pack(fill=tk.X, pady=5)
        self.query_entry.bind("<Return>", lambda e: self._send_query())
        
        # Кнопки
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X)
        
        self.send_button = ttk.Button(button_frame, text="Отправить", 
                                     command=self._send_query, state=tk.DISABLED)
        self.send_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Очистить чат", 
                  command=self._clear_chat).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Статистика", 
                  command=self._show_stats).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Очистить кеш", 
                  command=self._clear_cache).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Загрузить документы", 
                  command=self._load_documents).pack(side=tk.LEFT, padx=5)
        
    def _on_mode_change(self):
        """Обработка смены режима."""
        if self.pipeline:
            self.pipeline = None
            self.status_label.config(text="Не инициализировано", foreground="red")
            self.send_button.config(state=tk.DISABLED)
            self._add_system_message("Режим изменен. Требуется повторная инициализация.")
    
    def _initialize_pipeline(self):
        """Инициализация RAG pipeline."""
        mode = self.mode.get()
        
        # Проверка ключей
        if mode == "api":
            if not os.getenv("OPENAI_API_KEY"):
                messagebox.showerror("Ошибка", "OPENAI_API_KEY не установлен в .env файле")
                return
        else:
            if not os.getenv("GIGACHAT_AUTH_KEY") or not os.getenv("GIGACHAT_RQUID"):
                messagebox.showerror("Ошибка", "GIGACHAT_AUTH_KEY или GIGACHAT_RQUID не установлены в .env файле")
                return
        
        self.init_button.config(state=tk.DISABLED)
        self.status_label.config(text="Инициализация...", foreground="orange")
        self._add_system_message(f"Инициализация в режиме {'OpenAI API' if mode == 'api' else 'GigaChat'}...")
        
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self._init_pipeline_thread, args=(mode,))
        thread.daemon = True
        thread.start()
    
    def _init_pipeline_thread(self, mode):
        """Инициализация pipeline в отдельном потоке."""
        try:
            if mode == "api":
                from rag_pipeline import RAGPipeline
                self.pipeline = RAGPipeline(
                    collection_name="gui_api_collection",
                    cache_db_path="gui_api_cache.db",
                    data_file="../assistant_api/data/docs.txt",
                    model="gpt-4o-mini"
                )
            else:
                sys.path.insert(0, str(Path(__file__).parent.parent / 'assistant_giga'))
                from rag_pipeline import RAGPipeline
                self.pipeline = RAGPipeline(
                    collection_name="gui_giga_collection",
                    cache_db_path="gui_giga_cache.db",
                    data_file="../assistant_giga/data/docs.txt",
                    model="GigaChat"
                )
            
            self.root.after(0, self._init_success)
            
        except Exception as e:
            self.root.after(0, lambda: self._init_error(str(e)))
    
    def _init_success(self):
        """Успешная инициализация."""
        self.status_label.config(text="✓ Готов к работе", foreground="green")
        self.send_button.config(state=tk.NORMAL)
        self.init_button.config(state=tk.NORMAL)
        self._add_system_message("Система готова к работе!")
    
    def _init_error(self, error):
        """Ошибка инициализации."""
        self.status_label.config(text="Ошибка", foreground="red")
        self.init_button.config(state=tk.NORMAL)
        messagebox.showerror("Ошибка инициализации", f"Не удалось инициализировать систему:\n{error}")
    
    def _send_query(self):
        """Отправка запроса."""
        if not self.pipeline:
            messagebox.showwarning("Предупреждение", "Сначала инициализируйте систему")
            return
        
        query = self.query_entry.get().strip()
        if not query:
            return
        
        self.query_entry.delete(0, tk.END)
        self._add_user_message(query)
        
        self.send_button.config(state=tk.DISABLED)
        
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self._process_query_thread, args=(query,))
        thread.daemon = True
        thread.start()
    
    def _process_query_thread(self, query):
        """Обработка запроса в отдельном потоке."""
        try:
            result = self.pipeline.query(query)
            self.root.after(0, lambda: self._display_result(result))
        except Exception as e:
            self.root.after(0, lambda: self._query_error(str(e)))
    
    def _display_result(self, result):
        """Отображение результата."""
        answer = result['answer']
        from_cache = result.get('from_cache', False)
        
        if from_cache:
            self._add_assistant_message(answer, cache=True)
        else:
            self._add_assistant_message(answer)
        
        self.send_button.config(state=tk.NORMAL)
    
    def _query_error(self, error):
        """Ошибка обработки запроса."""
        self._add_system_message(f"Ошибка: {error}")
        self.send_button.config(state=tk.NORMAL)
    
    def _add_user_message(self, text):
        """Добавление сообщения пользователя."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n👤 Вы: ", "user")
        self.chat_display.insert(tk.END, f"{text}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _add_assistant_message(self, text, cache=False):
        """Добавление ответа ассистента."""
        self.chat_display.config(state=tk.NORMAL)
        if cache:
            self.chat_display.insert(tk.END, "\n🤖 Ассистент ", "assistant")
            self.chat_display.insert(tk.END, "(из кеша): ", "cache")
        else:
            self.chat_display.insert(tk.END, "\n🤖 Ассистент: ", "assistant")
        self.chat_display.insert(tk.END, f"{text}\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _add_system_message(self, text):
        """Добавление системного сообщения."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"\n💡 {text}\n", "system")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _clear_chat(self):
        """Очистка чата."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _show_stats(self):
        """Показать статистику."""
        if not self.pipeline:
            messagebox.showwarning("Предупреждение", "Сначала инициализируйте систему")
            return
        
        stats = self.pipeline.get_stats()
        
        stats_text = f"""
📊 СТАТИСТИКА СИСТЕМЫ

🗄️ Векторное хранилище:
   Коллекция: {stats['vector_store']['name']}
   Документов: {stats['vector_store']['count']}
   Директория: {stats['vector_store']['persist_directory']}

💾 Кеш:
   Записей: {stats['cache']['total_entries']}
   Размер БД: {stats['cache']['db_size_mb']:.2f} MB
   Первая запись: {stats['cache']['oldest_entry'] or 'N/A'}
   Последняя запись: {stats['cache']['newest_entry'] or 'N/A'}

🤖 Модель: {stats['model']}
🌐 Режим: {stats.get('mode', 'N/A')}
        """
        
        messagebox.showinfo("Статистика", stats_text)
    
    def _clear_cache(self):
        """Очистка кеша."""
        if not self.pipeline:
            messagebox.showwarning("Предупреждение", "Сначала инициализируйте систему")
            return
        
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить кеш?"):
            self.pipeline.cache.clear()
            self._add_system_message("Кеш очищен")
            messagebox.showinfo("Успех", "Кеш успешно очищен")
    
    def _load_documents(self):
        """Загрузка документов и создание эмбеддингов."""
        if not self.pipeline:
            messagebox.showwarning("Предупреждение", "Сначала инициализируйте систему")
            return
        
        # Выбор файла
        file_path = filedialog.askopenfilename(
            title="Выберите файл с документами",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Подтверждение
        current_count = self.pipeline.vector_store.collection.count()
        if current_count > 0:
            if not messagebox.askyesno(
                "Подтверждение",
                f"В векторном хранилище уже есть {current_count} документов.\n" +
                "Загрузка новых документов добавит их к существующим.\n" +
                "Продолжить?"
            ):
                return
        
        self._add_system_message(f"Загрузка документов из {os.path.basename(file_path)}...")
        
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self._load_documents_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _load_documents_thread(self, file_path):
        """Загрузка документов в отдельном потоке."""
        try:
            # Чтение файла
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Разбиение на чанки
            chunks = self.pipeline.vector_store._chunk_text(text)
            self.root.after(0, lambda: self._add_system_message(f"Текст разбит на {len(chunks)} чанков"))
            
            # Создание эмбеддингов и добавление в ChromaDB
            documents = []
            ids = []
            embeddings = []
            
            current_max_id = self.pipeline.vector_store.collection.count()
            
            for i, chunk in enumerate(chunks):
                embedding = self.pipeline.vector_store._create_embedding(chunk)
                documents.append(chunk)
                ids.append(f"doc_{current_max_id + i}")
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    self.root.after(0, lambda idx=i: self._add_system_message(
                        f"Обработано {idx + 1}/{len(chunks)} чанков"
                    ))
            
            # Добавление в ChromaDB
            self.pipeline.vector_store.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            
            self.root.after(0, lambda: self._load_documents_success(len(chunks)))
            
        except Exception as e:
            self.root.after(0, lambda: self._load_documents_error(str(e)))
    
    def _load_documents_success(self, count):
        """Успешная загрузка документов."""
        total = self.pipeline.vector_store.collection.count()
        self._add_system_message(f"✓ Загружено {count} документов. Всего в хранилище: {total}")
        messagebox.showinfo("Успех", f"Документы успешно загружены!\nДобавлено: {count}\nВсего: {total}")
    
    def _load_documents_error(self, error):
        """Ошибка загрузки документов."""
        self._add_system_message(f"✗ Ошибка загрузки: {error}")
        messagebox.showerror("Ошибка", f"Не удалось загрузить документы:\n{error}")


def main():
    """Запуск GUI приложения."""
    root = tk.Tk()
    app = RAGAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
