import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Шлях до файлу
file_path = '/Users/mariana.liakh/Downloads/netflix_titles_nov_2019.csv'

# Завантаження CSV у DataFrame
data = pd.read_csv(file_path)

# Перегляд перших кількох рядків
print(data.head())

# Перевірка на пропущені значення
print(data.isnull().sum())

# Заповнення пропущених значень
data['director'].fillna('Unknown', inplace=True)
data['country'].fillna('Unknown', inplace=True)

# Перетворення колонок типу дати
data['date_added'] = pd.to_datetime(data['date_added'])

# Нормалізація числових колонок
scaler = MinMaxScaler()
data['release_year_scaled'] = scaler.fit_transform(data[['release_year']])

# Перетворення тривалості в рядковий формат, якщо ще не рядок
data['duration'] = data['duration'].astype(str)

# Видалити 'min' і пробіли
data['duration'] = data['duration'].str.replace('min', '').str.strip()

# Створити маску для значень, що містять 'Season'
season_mask = data['duration'].str.contains('Season', na=False)

# Витягнути числа для 'Season' і перетворити інші значення на числовий формат
data.loc[season_mask, 'duration'] = data.loc[season_mask, 'duration'].str.extract(r'(\d+)').astype(float)

# Перетворити залишок на числовий формат
data['duration'] = pd.to_numeric(data['duration'], errors='coerce')

# Заповнення пропущених значень середнім значенням
data['duration'].fillna(data['duration'].mean(), inplace=True)

# Перевірка на пропущені значення після перетворення
print(data['duration'].isna().sum())  # Кількість пропущених значень

# Виведення перших кількох значень тривалості
print(data['duration'].head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Перетворення категоріальних даних у числовий формат
data['rating'] = data['rating'].astype('category').cat.codes
data['listed_in'] = data['listed_in'].astype('category').cat.codes

# Вибір характеристик та міток
X = data[['release_year', 'rating', 'duration']]  # Можна обрати інші
y = data['listed_in']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Тренування моделі
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Передбачення та оцінка
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

################

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def fill_missing_values(self):
        self.data['director'].fillna('Unknown', inplace=True)
        self.data['country'].fillna('Unknown', inplace=True)

    def preprocess(self):
        self.fill_missing_values()
        self.data['date_added'] = pd.to_datetime(self.data['date_added'])
        # Інші етапи обробки...

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestClassifier()

    def train(self):
        self.model.fit(self.X_train, self.y_train)
    # Інші методи для тестування та оцінки моделі
