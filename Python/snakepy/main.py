import pygame
from pygame.locals import *
from sys import exit
import time
import random

SIZE=34
time_rate=0.1
BG_COLOR=(103, 162, 245)
class Food:
    def __init__(self,parent_screen):
        self.app=pygame.image.load("resourse/food.jpg").convert()
        self.parent_screen=parent_screen
        self.x=SIZE*3
        self.y=SIZE*3
    def draw(self):  
        self.parent_screen.blit(self.app,(self.x,self.y))
        pygame.display.flip()
    def move(self):
        self.x=random.randint(0,24)*SIZE
        self.y=random.randint(0,20)*SIZE

class Snake:
    def __init__(self,parent_screen,length):
        self.length=length
        self.parent_screen=parent_screen
        self.block=pygame.image.load("resourse/head.jpg").convert()
        self.x=[SIZE]*length
        self.y=[SIZE]*length
        self.direction='up'
    def draw(self):
        self.parent_screen.fill(BG_COLOR)
        for i in range(self.length):    
            self.parent_screen.blit(self.block,(self.x[i],self.y[i]))
        pygame.display.flip()
    def move_up(self):
        self.direction='up'
    def move_down(self):
        self.direction='down'
    def move_right(self):
        self.direction='right'
    def move_left(self):
        self.direction='left'
    def cont_drxn(self):
        for i in range(self.length-1,0,-1):
            self.x[i]=self.x[i-1]
            self.y[i]=self.y[i-1]
            
        if self.direction=='up':
            self.y[0]-=SIZE
        if self.direction=='down':
            self.y[0]+=SIZE
        if self.direction=='left':
            self.x[0]-=SIZE
        if self.direction=='right':
            self.x[0]+=SIZE
        self.draw()
    
    def inc_length(self):
        self.length+=1
        self.x.append(-1)
        self.y.append(-1)
        
        

class Game:
    def __init__(self):
        pygame.init()
        self.surface=pygame.display.set_mode((1000,800))
        self.surface.fill((103, 162, 245))
        self.snake=Snake(self.surface,1)
        self.snake.draw()
        self.food=Food(self.surface)
        self.food.draw()
    
    def is_collison(self,x1,y1,x2,y2):
        if x1>=x2 and x1<x2+SIZE:
            if y1>=y2 and y1<y2+SIZE:
                return True
    
        
    
    def play(self):
        global time_rate
        self.snake.cont_drxn()
        self.food.draw()
        self.display_score()
        pygame.display.flip()
        if self.is_collison(self.snake.x[0],self.snake.y[0],self.food.x,self.food.y):
            self.food.move()
            self.snake.inc_length()
            
        for i in range(3,self.snake.length):
            if self.is_collison(self.snake.x[0],self.snake.y[0],self.snake.x[i],self.snake.y[i]):
                raise "Game Over"
                
    
    def display_score(self):
       font=pygame.font.SysFont('arial', 30)
       score=font.render(f"Score:{self.snake.length}",True,(200,200,200))
       self.surface.blit(score,(800,10))
        
    def show_game_over(self):
        self.surface.fill(BG_COLOR)
        font=pygame.font.SysFont('arial', 30)
        line1=font.render(f"Game khtm paissa hzm ... Your score is:{self.snake.length}",True,(255,255,255))
        self.surface.blit(line1,(200,300))
        line2=font.render("To play Press Enter \nTo end press Escape", True, (255,255,255))
        self.surface.blit(line2,(200,350))
        pygame.display.flip()
        
        
    def reset(self):
        self.snake=Snake(self.surface,1)
        self.food=Food(self.surface)
        
    def run(self):
        running=True
        pause=False
        while running:
            for event in pygame.event.get():
                if event.type==KEYDOWN:
                    if event.key==K_ESCAPE:
                        pygame.quit()
                        exit()
                    if event.key==K_UP:
                        self.snake.move_up()
                    if event.key==K_RETURN:
                        pause=False
                    if not pause:
                        if event.key==K_DOWN:
                            self.snake.move_down()
                        if event.key==K_LEFT:
                            self.snake.move_left()
                        if event.key==K_RIGHT:
                            self.snake.move_right()
                elif event.type==pygame.QUIT:
                    pygame.quit()
                    exit()
            try:
                if not pause:
                    self.play()
            except Exception as e:
                self.show_game_over()
                pause=True
                self.reset()
            time.sleep(time_rate)




if __name__=="__main__":
    game=Game()
    game.run()

    
