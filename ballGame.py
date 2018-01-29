import sys, pygame
import tensorflow as tf
import random

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
BEST_NETWORK_SPLIT=0.75
MAX_GENS=20

#Define size
DISPLAY_SIZE=(1024, 768)
thickness_def=20
grid_def=600
pad_def=100


currentGeneration=1

ball_pos=dotdict({"x" : int(DISPLAY_SIZE[0]/2), "y" : int(DISPLAY_SIZE[1]/2)})
pad_pos= dotdict({"x" : int(DISPLAY_SIZE[0]/2), "y" : int((DISPLAY_SIZE[1]-grid_def)/2)+grid_def - thickness_def})
speed=dotdict({"x" : 4, "y" : 4})

pygame.init()
screen= pygame.display.set_mode(DISPLAY_SIZE)
pygame.display.set_caption("Pong Genetic Algorithm")

pygame.font.init() # you have to call this at the start, 
myfont = pygame.font.SysFont('Comic Sans MS', 25)


networks=[]
pads=[]
no_networks=4
score=0

for _ in range(no_networks):
    curr=dotdict({})
    curr.weights= tf.random_normal([6,1], mean=0, stddev=1)
    curr.bias= tf.random_normal([1], mean=0, stddev=1)
    curr.score=0
    networks.append(curr)


for _ in range(no_networks):
    curr=dotdict({})
    curr.color= (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255) ) 
    curr.x=int(DISPLAY_SIZE[0]/2)
    curr.y=int((DISPLAY_SIZE[1]-grid_def)/2)+grid_def - thickness_def
    curr.dead=0
    pads.append(curr)

sess = tf.InteractiveSession()
print(sess.run(networks[0].weights))
#sess=tf.Session()
#game infor is ball x,y pos, pad x,y pos, ball speed l,r
#game_info_ = tf.placeholder(tf.float32, [6, 1])  #input is one value
#W = tf.Variable(tf.zeros([6, 1]))
#b = tf.Variable(tf.zeros([1]))
#y = tf.matmul(x, W) + b

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
        new_networks.append(new_obj )

    #print("new networks",len(new_networks), new_networks)
    #breed networks
    random.shuffle(new_networks)
    for n in range(0,len(new_networks), 2):
        #print(n)
        weight1=sess.run(new_networks[n].weights)
        weight2=sess.run(new_networks[n+1].weights)
        #print("Weights to be rearranged and bred: ", weight1, weight2)
        weight1= [ x[0] for x in weight1]
        weight2= [ x[0] for x in weight2]
        #print("Weights fixed to be rearranged and bred: ", weight1, weight2)

        mid= int(len(weight1)/2)
        new_networks[n].weights= tf.constant(weight1[:mid]+ weight2[mid:], shape=[6,1])
        new_networks[n+1].weights = tf.constant(weight2[:mid]+ weight1[mid:], shape=[6,1])
        
    

    #random mutation- 0.15 chance
    for n in range(len(new_networks)):
        weights_replace=[]
        curr_weights=sess.run(new_networks[n].weights)
        #print("Weights to be mutated: ", curr_weights)

        for w in range(len(curr_weights)):
            if( random.uniform(0.0,1.0) >=MUTATE_PROB):
                weights_replace.append(curr_weights[w][0]+random.uniform(-MUTATE_THRESH, MUTATE_THRESH))
            else:
                weights_replace.append(curr_weights[w][0])
        new_networks[n].weights= tf.constant(weights_replace,shape=[6,1] )
        if( random.uniform(0.0,1.0) >=MUTATE_PROB):
            new_networks[n].bias= tf.add( new_networks[n].bias, tf.constant(random.uniform(-MUTATE_THRESH, MUTATE_THRESH)))     


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

    textsurface2 = myfont.render('Score '+ str(score), True, WHITE)
    screen.blit(textsurface2,(500,50))

    drawn_pads={}
    for i in range(len(pads)):
        p=pads[i]
        if(not p.dead ):
            if p.x<= DISPLAY_SIZE[0] and p.x>=0:
                pad=pygame.draw.rect(screen, p.color, [p.x,p.y,pad_def,thickness_def])
                drawn_pads[i]=pad
            elif p.x>=DISPLAY_SIZE[0]:
                pad=pygame.draw.rect(screen, p.color, [DISPLAY_SIZE[0]-pad_def,p.y,pad_def,thickness_def])
                drawn_pads[i]=pad
            else:
                pad=pygame.draw.rect(screen, p.color, [0,p.y,pad_def,thickness_def])
                drawn_pads[i]=pad

    pygame.display.flip()

    if(count==4):
        count=0
        if(ball.colliderect(grid.top)):
            speed.y=-speed.y
            print("Hit top reversing direction")
        if(ball.colliderect(grid.left) or ball.colliderect(grid.right)):
            speed.x=-speed.x
            print("Hit left/right reversing direction")

        for n in range(len(networks)):
            if not pads[n].dead:
                input_arr=[[ball_pos.x, ball_pos.y, pads[n].x, pads[n].y, speed.x, speed.y]]
                move= tf.matmul(tf.convert_to_tensor(input_arr, dtype=tf.float32) , networks[n].weights)+ networks[n].bias
                move=sess.run(move)
                pads[n].x+=move[0][0]

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
                    networks[d].score-=abs(pads[d].x-ball_pos.x)/(grid_def-(thickness_def*2))
            print("Number of pads still active:",  sum([not p.dead for p in pads]) )

        if(not any([not p.dead for p in pads])):
            print("All pads have failed")
            if(currentGeneration <MAX_GENS):
                ball_pos.y=int(DISPLAY_SIZE[1]/2)
                evolveGraphs()
                resetScores()
                currentGeneration+=1

    count+=1
    ball_pos.x+=speed.x
    ball_pos.y+=speed.y


    #if(ball.colliderect(grid.bottom)):
    #    print("Game has failed")








