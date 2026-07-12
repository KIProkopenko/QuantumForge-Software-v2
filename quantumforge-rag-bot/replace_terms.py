import json
import os
import re
from pathlib import Path

def load_terms_map(terms_file='terms_map.json'):
    """Загружает словарь замен из JSON-файла"""
    with open(terms_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def replace_terms_in_text(text, terms_map):
    """Заменяет все термины в тексте согласно словарю"""
    # Сортируем термины по длине (длинные сначала, чтобы избежать частичных замен)
    sorted_terms = sorted(terms_map.keys(), key=len, reverse=True)
    
    for original_term in sorted_terms:
        replacement = terms_map[original_term]
        # Используем regex с word boundaries для точной замены
        pattern = r'\b' + re.escape(original_term) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def process_knowledge_base(kb_dir='knowledge_base', terms_file='terms_map.json'):
    """Обрабатывает все файлы в папке knowledge_base"""
    terms_map = load_terms_map(terms_file)
    kb_path = Path(kb_dir)
    
    if not kb_path.exists():
        print(f"Папка {kb_dir} не найдена!")
        return
    
    processed_files = 0
    for file_path in kb_path.glob('*.*'):
        if file_path.suffix in ['.md', '.txt']:
            print(f"Обрабатываю: {file_path.name}")
            
            # Читаем файл
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Заменяем термины
            new_content = replace_terms_in_text(content, terms_map)
            
            # Перезаписываем файл
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            processed_files += 1
    
    print(f"\nОбработано файлов: {processed_files}")

if __name__ == '__main__':
    print("Запуск скрипта подмены терминов...")
    process_knowledge_base()
    print("Готово!")