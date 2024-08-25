import pickle
import pygame
import time
import math
import os
import neat
from utils import *

pygame.font.init()

GRASS = scale_image(pygame.image.load(os.path.join("imgs", "grass.jpg")), 2.5)
TRACK = scale_image(pygame.image.load(os.path.join("imgs", "track.png")), 0.9)
TRACK_BORDER = scale_image(pygame.image.load(os.path.join("imgs", "track-border.png")),0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
FINISH_LINE = pygame.image.load(os.path.join("imgs", "finish.png"))
FINISH_LINE_POSITION =(111, 250)
FINISH_LINE_MASK = pygame.mask.from_surface(FINISH_LINE)
CAR = scale_image(pygame.image.load(os.path.join("imgs", Img_select())), 0.55)
DRAW_LINES = False
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Racing Game AI")

gen =0

class Car:
    IMG =  CAR
    START_POS = (165, 200)
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel =max_vel
        self.vel = 1
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.prev_x = self.x
        self.prev_y = self.y
        self.distance_traveled = 0
        self.stuck_frames = 0
        self.rotation_frames = 0
        self.acc =0
        self.total_distances =0


    def rotate(self, left=False, right = False):
        if left:
            self.angle += self.rotation_vel
            self.rotation_frames += 1
        elif right:
            self.angle -= self.rotation_vel
            self.rotation_frames += 1

    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
    
    def move_forward(self):
        self.vel = min(self.vel + self.acceleration , self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration , -self.max_vel/2)
        self.move()

    def Go(self):
        if self.acc > 0:
            self.acc = 4    
        self.vel = min(self.vel + self.acceleration , self.max_vel)
        radians = math.radians(self.angle)
        vertical = math.cos(radians)*self.vel
        horizontal = math.sin(radians)*self.vel

        self.y -= vertical
        self.x -= horizontal

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians)*self.vel
        horizontal = math.sin(radians)*self.vel

        self.y -= vertical
        self.x -= horizontal

        distance = math.sqrt((self.x - self.prev_x)**2 + (self.y - self.prev_y)**2)
        self.distance_traveled += distance
        self.total_distances += distance
        self.prev_x = self.x
        self.prev_y = self.y

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x-x), int(self.y-y))
        point = mask.overlap(car_mask, offset)
        return point

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()
    
    def bounces(self):
        self.vel = -self.vel
        self.move()




def draw(win, images, cars, gen):
    if gen == 0:
        gen = 1
    for img, pos in images:
        win.blit(img, pos)

    for car in cars:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (car.x+car.img.get_width()/2, car.y+car.img.get_height()/2),(TRACK_BORDER_MASK.get_width()/2, TRACK_BORDER_MASK.get_height()/2), 0.9)
            except:
                pass
        car.draw(win)
    
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))
    pygame.display.update()
    

def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a]:
        player_car.rotate(left=True)

    if keys[pygame.K_d]:
        player_car.rotate(right=True)
    if keys[pygame.K_w]:
        moved = True
        player_car.move_forward()
    if keys[pygame.K_s]:
        moved =True
        player_car.move_backward()
    
    if not moved:
        player_car.reduce_speed()


def main(genomes, config):
    global WIN, gen
    win = WIN

    nets = []
    cars = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(4,4))
        ge.append(genome)

    result = 0
    FPS =30
    run = True
    clock = pygame.time.Clock()
    images = [(GRASS,(0,0)), (TRACK, (0,0)), (FINISH_LINE, FINISH_LINE_POSITION),(TRACK_BORDER, (0,0))]

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                        run = False

        for x, car in enumerate(cars):
            car.Go()
            # ge[x].fitness += 0.0001

            mask_value = calculate_distance_to_border(car, TRACK_BORDER_MASK)
            car_winingline = calculate_distance_to_finish(car, FINISH_LINE_POSITION)
            p1 = calculate_distance_to_finish(car,(10, 250))
            p2 = calculate_distance_to_finish(car,(135, 600))
            p3 = calculate_distance_to_finish(car,(550, 600))
            p4 = calculate_distance_to_finish(car,(690, 600))
            p5 = calculate_distance_to_finish(car,(690, 150))
            p6 = calculate_distance_to_finish(car,(225, 170))
            output = nets[cars.index(car)].activate((car.x, car.y,  car.angle,mask_value,car_winingline,p1,p2,p3,p4,p5,p6))

            if output[0] > 0.5:
                car.rotate(left=True)
                ge[x].fitness += 1

            if output[1] > 0.5:
                car.rotate(right=True)
                ge[x].fitness += 1

            if car.collide(FINISH_LINE_MASK, x=10, y=250)!= None:
                ge[x].fitness += 0.1
            
            if car.collide(FINISH_LINE_MASK, x=120, y=580)!= None:
                ge[x].fitness += 0.1

            if car.collide(FINISH_LINE_MASK, x=550, y=600)!= None:
                ge[x].fitness += 0.1
            
            if car.collide(FINISH_LINE_MASK, x=690, y=600)!= None:
                ge[x].fitness += 0.1

            if car.collide(FINISH_LINE_MASK, x=690, y=150)!= None:
                ge[x].fitness += 0.1
            
            if car.collide(FINISH_LINE_MASK, x=225, y=170)!= None:
                ge[x].fitness += 0.1

            if car.distance_traveled >= 10:
                ge[x].fitness += 0.1
                car.distance_traveled -= 10


        for car in cars:
            if car.collide(TRACK_BORDER_MASK) != None:
                ge[cars.index(car)].fitness -= 10
                nets.pop(cars.index(car))
                ge.pop(cars.index(car))
                cars.pop(cars.index(car))
            
            move_player(car)
            

            finsh_check =car.collide(FINISH_LINE_MASK, *FINISH_LINE_POSITION)
            if finsh_check != None:
                ge[cars.index(car)].fitness += 10
                result += 1
                if result == 2:
                    pickle.dump(nets[0],open("best.pickle", "wb"))
                    break 
                # if finsh_check[1] == 0:
                #         car.bounces()
                #         ge[cars.index(car)].fitness -= 1
                #         nets.pop(cars.index(car))
                #         ge.pop(cars.index(car))
                #         cars.pop(cars.index(car))
                # else:                  
                #     pickle.dump(nets[0],open("best.pickle", "wb"))
                #     break        

        if len(cars) == 0:
            gen += 1  # Increment the generation counter
            return

        draw(WIN, images, cars, gen)
        pygame.display.update()
            
    
        
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 100000000000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
