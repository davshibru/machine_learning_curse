
# Шаг 1 импорт заранее подготовленых библеотек и настроек
from lab_9_2.utils import *

# Шаг 2 создание окна для рисования

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drawing Program")

# Шаг 3 подготовка функции для обозначения сетки 8х8

def init_grid(rows, cols, color):
    grid = []

    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(color)

    return grid

# Шаг 4 подготовка функции для рисвания сетки на окне для рисования

def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(win, pixel, (j * PIXEL_SIZE, i *
                                          PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    if DRAW_GRID_LINES:
        for i in range(ROWS + 1):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE),
                             (WIDTH, i * PIXEL_SIZE))

        for i in range(COLS + 1):
            pygame.draw.line(win, BLACK, (i * PIXEL_SIZE, 0),
                             (i * PIXEL_SIZE, HEIGHT - TOOLBAR_HEIGHT))


# Шаг 5 создание функции для рисования элементов на окне для рисования

def draw(win, grid, buttons, s):
    win.fill(BG_COLOR)
    # использования функции для рисования сетки
    draw_grid(win, grid)

    # рисование кнопок на окне
    for button in buttons:
        button.draw(win)

    # вывод на экран результата распознания рукопистных цифр
    x = 320
    y = HEIGHT - TOOLBAR_HEIGHT/2 - 25
    wight = 50
    for i in s:
        draw_numbers(win, x, y, wight, i)
        x += 60
        wight -= 12

    pygame.display.update()

# Шаг 6 подготовка функции для вывода результата распознания на экран
def draw_numbers(win, x, y, wight, s):
    pygame.draw.rect(
        win, WHITE, (x, y, wight, wight))
    pygame.draw.rect(
        win, BLACK, (x, y, wight, wight), 2)

    button_font = get_font(22)
    text_surface = button_font.render(s, 1, BLACK)
    win.blit(text_surface, (x + wight /
                            2 - text_surface.get_width() / 2,
                            y + wight / 2 - text_surface.get_height() / 2))

# Шаг 7 создания функции которая определяет координаты пикселя который был закрашен
def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE

    if row >= ROWS:
        raise IndexError

    return row, col

# Шаг 8 подготовка переменных

# переменная для определения работы программы
run = True
# переменная для установки частоты обнавления экрана программы
clock = pygame.time.Clock()
# переменная которая хранит сетку в которой будет производится рисования
grid = init_grid(ROWS, COLS, BG_COLOR)
# переменная которая хранит цвет кисти для рисования
drawing_color = BLACK
# переменная которая хранит в массиве информацию нарисованную пользователем
predict_list = []

# переменная хранящая высоту на которой будет расположен toolbar показывающий кнопки и результат распознания рукописных цифр
button_y = HEIGHT - TOOLBAR_HEIGHT/2 - 25

# переменная хранящая кнопки в toolbar
buttons = [
    # кнопка кисти
    Button(10, button_y, 50, 50, BLACK),
    # кнопка ластика
    Button(70, button_y, 50, 50, WHITE, "Erase", BLACK),
    # кнопка "стереть все"
    Button(130, button_y, 50, 50, WHITE, "Clear", BLACK),
]


# Шаг 9 запуск бесконечного цикла, который остановится когда переменная run меняет значение на False
while run:

    # установка чатоту обнавления экрана окна (30)
    clock.tick(FPS)

    # цикл который проверяет переберает событи происходящие в окне
    for event in pygame.event.get():

        # если пользователь закрывает окно переменная run меняется на False
        if event.type == pygame.QUIT:
            run = False

        # реагирование на нажатие кнопок в окне
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()

            try:
                row, col = get_row_col_from_pos(pos)
                grid[row][col] = drawing_color
            except IndexError:
                for button in buttons:
                    if not button.clicked(pos):
                        continue

                    drawing_color = button.color
                    if button.text == "Clear":
                        grid = init_grid(ROWS, COLS, BG_COLOR)
                        drawing_color = BLACK

    # очищение переменной predict_list
    predict_list = []

    # заполнение массива predict_list данными из сетки
    for i in grid:
        for l in i:
            if l == (255, 255, 255):
                predict_list.append(0.0)
            else:
                predict_list.append(15.0)

    # обнавление экрана вызовам метода draw
    draw(WIN, grid, buttons, check(predict_list))

# Шаг 10 остановка pygame
pygame.quit()



