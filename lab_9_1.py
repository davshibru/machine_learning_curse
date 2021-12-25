from lab_9_2.utils import *

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drawing Program")


def init_grid(rows, cols, color):
    grid = []

    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(color)

    return grid


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



def draw(win, grid, buttons, s):
    win.fill(BG_COLOR)
    draw_grid(win, grid)

    for button in buttons:
        button.draw(win)

    x = 320
    y = HEIGHT - TOOLBAR_HEIGHT/2 - 25
    wight = 50
    for i in s:
        draw_numbers(win, x, y, wight, i)
        x += 60
        wight -= 12

    pygame.display.update()


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

def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE

    if row >= ROWS:
        raise IndexError

    return row, col


run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BG_COLOR)
drawing_color = BLACK
predict_list = []

button_y = HEIGHT - TOOLBAR_HEIGHT/2 - 25
buttons = [
    Button(10, button_y, 50, 50, BLACK),
    Button(70, button_y, 50, 50, WHITE, "Erase", BLACK),
    Button(130, button_y, 50, 50, WHITE, "Clear", BLACK),
]



while run:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

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

    predict_list = []

    for i in grid:
        for l in i:
            if l == (255, 255, 255):
                predict_list.append(0.0)
            else:
                predict_list.append(15.0)


    draw(WIN, grid, buttons, check(predict_list))

pygame.quit()