# Шаг 16 подготовка библиотек
import pygame

# Шаг 17 запуск pygame
pygame.init()
pygame.font.init()

# Шаг 18 Подгатовка цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 255, 0)
GREEN = (0, 0, 255)

# Шаг 19 установка чатоту обнавления экрана
FPS = 30

# Шаг 20 установка размера экрана
WIDTH, HEIGHT = 600, 700

# Шаг 21 установка количества колонок и строк в сетке для рисования
ROWS = COLS = 8

# Шаг 22 установка высоты размера toolbar
TOOLBAR_HEIGHT = HEIGHT - WIDTH

# Шаг 23 установка значения пиксиля который будет закрашиваться в сетке
PIXEL_SIZE = WIDTH // COLS

# Шаг 24 установка заднего фона
BG_COLOR = WHITE

# Шаг 25 создание переменной котороя отвечает за то чтобы сетка была или не была наресована
DRAW_GRID_LINES = True

# Шаг 26 получение шрифта
def get_font(size):
    return pygame.font.SysFont("comicsans", size)