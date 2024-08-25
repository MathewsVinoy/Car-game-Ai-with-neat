import math
import random
import pygame

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

def Img_select():
        name = ["green-car.png", "red-car.png", "grey-car.png","purple-car.png", "white-car.png"]
        name = random.choice(name)
        return name

def blit_rotate_center(win, img, top_left, angle):
    rotated_image = pygame.transform.rotate(img, angle)
    new_rect = rotated_image.get_rect(center=top_left)
    win.blit(rotated_image, new_rect.topleft)

def calculate_distance_to_border(car, track_border_mask):
    car_mask = pygame.mask.from_surface(car.img)  # Create a mask from the car's image
    overlap = track_border_mask.overlap(car_mask, (car.x, car.y))  # Pass the car's mask and position
    if overlap:
        overlap_x, overlap_y = overlap
        car_center_x, car_center_y = car.x + car.img.get_width() // 2, car.y + car.img.get_height() // 2
        distance = ((car_center_x - overlap_x) ** 2 + (car_center_y - overlap_y) ** 2) ** 0.5
    else:
        distance = float('inf')  # or some large value if there's no overlap
    return distance

def calculate_distance_to_finish(car, finish_line_position):
    x_diff = car.x - finish_line_position[0]
    y_diff = car.y - finish_line_position[1]
    distance = math.sqrt(x_diff**2 + y_diff**2)
    return distance