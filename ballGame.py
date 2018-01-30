import sys, pygame
#import tensorflow as tf
import random
import numpy as np

from copy import deepcopy


#Declaring own version of __setattr__ etc to allow dot notation
#__setattr__ is class method called by setattr builtin method, that is if __setattr__ is defined in given class
# can override __setatrr__ to provide obj like access to someother data structure
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get #.item
    __setattr__ = dict.__setitem__ #.item =
    __delattr__ = dict.__delitem__ 

# Define the colors rgb
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

MUTATE_THRESH=0.1
MUTATE_PROB=0.9
INITIAL_DEV=1
BEST_NETWORK_SPLIT=0.6
MAX_GENS=20
SCORE_SUCCESS=3


#Define size
DISPLAY_SIZE=(1024, 768)
thickness_def=20
grid_def=600
pad_def=100




ball_pos=dotdict({"x" : int(DISPLAY_SIZE[0]/2), "y" : int(DISPLAY_SIZE[1]/2)})
pad_pos= dotdict({"x" : int(DISPLAY_SIZE[0]/2), "y" : int((DISPLAY_SIZE[1]-grid_def)/2)+grid_def - thickness_def})
speed=dotdict({"x" : 3, "y" : 5})

pygame.init()
screen= pygame.display.set_mode(DISPLAY_SIZE)
pygame.display.set_caption("Pong Genetic Algorithm")

pygame.font.init() # you have to call this at the start, 
myfont = pygame.font.SysFont('Comic Sans MS', 25)


networks=[]
pastGens=[]
pads=[]
no_networks=20
score=0
currentGeneration=1

def resetNetworks():
    global no_networks, networks, pads, currentGeneration, score
    networks=[]
    pads=[]
    score=0
    
    currentGeneration=1
    ball_pos.x=int(DISPLAY_SIZE[0]/2)
    ball_pos.y=int(DISPLAY_SIZE[1]/2)

    for _ in range(no_networks):
        curr=dotdict({})
        #curr.weights= tf.random_normal([6,1], mean=0, stddev=1)
        curr.weights= np.random.normal(0,INITIAL_DEV, (6)) #mean, stddev, shape
        curr.bias= np.random.normal(0,INITIAL_DEV)
        curr.score=0
        networks.append(curr)


    for _ in range(no_networks):
        curr=dotdict({})
        curr.color= (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255) ) 
        curr.x=int(DISPLAY_SIZE[0]/2)
        curr.y=int((DISPLAY_SIZE[1]-grid_def)/2)+grid_def - thickness_def
        curr.dead=0
        pads.append(curr)

resetNetworks()

def resetScores():
    global networks, pads
    for n in range(len(networks)):
        networks[n].score=0
    for p in pads:
        p.x, p.y= int(DISPLAY_SIZE[0]/2), int((DISPLAY_SIZE[1]-grid_def)/2)+grid_def - thickness_def
        p.dead=0

def evolveGraphs():
    global networks
    #Create array new_networks containing top + some random network configs
    
    networks.sort(key= lambda x: x.score)
    networks.reverse()
    print("Best score at generation ", currentGeneration, "is: ", networks[0].score)
    print("Evolving weights\n\n")
    #print("sorted networks", networks)
    new_networks=networks[0: int(len(networks)*BEST_NETWORK_SPLIT)]
    
    for _ in range(len(networks)-len(new_networks)):
        temp=networks[random.randrange(len(networks))]
        new_obj= dotdict({})
        new_obj.weights= temp.weights
        new_obj.bias= temp.bias
        new_obj.score=0
        new_networks.append(new_obj)

    #print("new networks",len(new_networks), new_networks)
    #breed networks
    random.shuffle(new_networks)
    for n in range(0,len(new_networks), 2):
        #print(n)
        weight1=new_networks[n].weights
        weight2=new_networks[n+1].weights
        #print("Weights to be rearranged and bred: ", weight1, weight2)
        #weight1= [ x[0] for x in weight1]
        #weight2= [ x[0] for x in weight2]
        #print("Weights fixed to be rearranged and bred: ", weight1, weight2)

        mid= int(len(weight1)/2)
        new_networks[n].weights= np.concatenate((weight1[:mid], weight2[mid:]), axis=0) 
        new_networks[n+1].weights = np.concatenate((weight2[:mid], weight1[mid:]), axis=0) 
        #if( random.uniform(0.0,1.0) >=0.8):
        #    new_networks[n].bias, new_networks[n+1].bias=new_networks[n+1].bias, new_networks[n].bias
    

    # random mutation - 0.15 chance
    for n in range(len(new_networks)):
        weights_replace=[]
        curr_weights=new_networks[n].weights
        #print("Weights to be mutated: " , curr_weights)

        for w in range(len(curr_weights)):
            if( random.uniform(0.0,1.0) >=MUTATE_PROB):
                weights_replace.append(curr_weights[w]+random.uniform(-MUTATE_THRESH, MUTATE_THRESH))
            else:
                weights_replace.append(curr_weights[w])
        new_networks[n].weights= weights_replace
        if( random.uniform(0.0,1.0) >=MUTATE_PROB):
            new_networks[n].bias= new_networks[n].bias + random.uniform(-MUTATE_THRESH, MUTATE_THRESH)    

    networks=new_networks


#tf.global_variables_initializer().run()

def is_collided_with(self, sprite):
    return self.rect.colliderect(sprite.rect)

def draw_grid(thickness=thickness_def, size=grid_def):

    x_corner= int((DISPLAY_SIZE[0]-size)/2)
    y_corner= int((DISPLAY_SIZE[1]-size)/2)
    left=pygame.draw.rect(screen, WHITE, [x_corner,y_corner,thickness,size])
    top=pygame.draw.rect(screen, WHITE, [x_corner,y_corner,size,thickness])
    #pygame.draw.rect(screen, WHITE, [100,600+width,600,width])
    right=pygame.draw.rect(screen, WHITE, [size-thickness+x_corner,y_corner,thickness,size])
    bottom=pygame.draw.rect(screen, BLACK, [x_corner,y_corner+size,size,thickness])
    pad_zone=pygame.draw.rect(screen, BLACK, [x_corner+thickness,y_corner+size-thickness,size-(2*thickness),thickness])
    return {"top": top, "bottom": bottom, "left": left, "right": right, "pad_zone":pad_zone}

count=0
while(1):
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)
    grid=dotdict(draw_grid())
    ball=pygame.draw.circle(screen, WHITE, (ball_pos.x, ball_pos.y), 10, 0)
    textsurface = myfont.render('Generation: '+ str(currentGeneration), True, WHITE)
    screen.blit(textsurface,(50,50))

    textsurface2 = myfont.render('Score '+ str(score) + "/"+ str(SCORE_SUCCESS), True, WHITE)
    screen.blit(textsurface2,(480,50))

    textsurface3 = myfont.render('Successful Gen', True, WHITE)
    screen.blit(textsurface3,(830,50))

    pastResults= pastGens if len(pastGens)<=10 else pastGens[len(pastGens)-10:]
    for i, gen in enumerate(pastResults):
        textsurface3 = myfont.render('Test '+ str(i)+ ": "+ str(gen), True, WHITE)
        screen.blit(textsurface3,(830,50+(i+1)*40)) 

    drawn_pads={}
    for i in range(len(pads)):
        p=pads[i]
        
        if(not p.dead):
            if p.x< (DISPLAY_SIZE[0]-pad_def) and p.x>=0:
                pad=pygame.draw.rect(screen, p.color, [p.x,p.y,pad_def,thickness_def])
                drawn_pads[i]=pad
            #Off right of screen
            elif p.x>= (DISPLAY_SIZE[0]-pad_def):
                pad=pygame.draw.rect(screen, p.color, [DISPLAY_SIZE[0]-pad_def,p.y,pad_def,thickness_def])
                drawn_pads[i]=pad
            #Off left of screen
            else:
                pad=pygame.draw.rect(screen, p.color, [0,p.y,pad_def,thickness_def])
                drawn_pads[i]=pad

    pygame.display.flip()

    if(count==3):
        count=0
        if(ball.colliderect(grid.top)):
            speed.y=-speed.y
            print("Hit top reversing direction")
        if(ball.colliderect(grid.left) or ball.colliderect(grid.right)):
            speed.x=-speed.x
            print("Hit left/right reversing direction")

        for n in range(len(networks)):
            if not pads[n].dead:
                input_arr=[ball_pos.x, ball_pos.y, pads[n].x, pads[n].y, speed.x, speed.y]
                #print("input arr ",np.array(input_arr))
                #print(networks[n].weights, np.reshape(np.array(input_arr),(1,6)))
                #print(np.sum(np.multiply(networks[n].weights, np.reshape(np.array(input_arr), (1,6) ) ) ))
                #print(networks[n].bias)
                move= np.sum(np.multiply(networks[n].weights, np.reshape(np.array(input_arr), (1,6) ) )) + networks[n].bias
                pads[n].x+=move

        if(ball.colliderect(grid.pad_zone)):
            print("Should be hitting paddle")
            for d in drawn_pads.keys():
                if(ball.colliderect(drawn_pads[d])):
                    print("Pad ",d, " successfully hit")
                    networks[d].score+=1
                    if(speed.y>0):
                        speed.y=-speed.y
                        score+=1
                else:
                    pads[d].dead=1
                    networks[d].score-=abs(pads[d].x-ball_pos.x)/(grid_def)
            print("Number of pads still active:",  sum([not p.dead for p in pads]) )

        if(not any([not p.dead for p in pads])):
            print("All pads have failed")
            if(currentGeneration <MAX_GENS):
                ball_pos.y=int(DISPLAY_SIZE[1]/2)
                evolveGraphs()
                resetScores()
                currentGeneration+=1
                score=0
            else:
                pastGens.append("Failed")
                resetNetworks()
        
        if(score>=SCORE_SUCCESS):
            pastGens.append(currentGeneration)
            resetNetworks()

            

    count+=1
    ball_pos.x+=speed.x
    ball_pos.y+=speed.y


    #if(ball.colliderect(grid.bottom)):
    #    print("Game has failed")








