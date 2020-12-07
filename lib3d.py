# A mini high level wrapper for python-OpenGL written by Divam Gupta  

import pygame as pg
from pygame.locals import *
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

from OpenGL.GLUT import *

import numpy as np

cubeVertices = ((1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1),(-1,1,1),(-1,-1,-1),(-1,-1,1),(-1,1,-1))
cubeEdges = ((0,1),(0,3),(0,4),(1,2),(1,7),(2,5),(2,3),(3,6),(4,6),(4,7),(5,6),(5,7))
cubeQuads = ((0,3,6,4),(2,5,6,3),(1,2,5,7),(1,0,4,7),(7,4,6,5),(2,3,0,1))


def init_gl():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.1, 1.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    glDepthFunc(GL_LEQUAL)



def wireCube():
    glBegin(GL_LINES)
    for cubeEdge in cubeEdges:
        for cubeVertex in cubeEdge:
            glVertex3fv(cubeVertices[cubeVertex])
    glEnd()

def solidCube():
    glBegin(GL_QUADS)
    for cubeQuad in cubeQuads:
        for cubeVertex in cubeQuad:
            glVertex3fv(cubeVertices[cubeVertex])

    # sphere = gluNewQuadric()
    # gluSphere(sphere, 0.3 , 100, 100)
    # gluCylinder(sphere,0.1 ,0.1 ,3.0 ,32,32)


    glEnd()



class Scene(object):
    """docstring for Scene"""
    def __init__(self):
        pg.init()
        display = (400, 400)
        pg.display.set_mode(display, DOUBLEBUF|OPENGL)

        init_gl()



        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)
        

        #glLight(GL_LIGHT0, GL_POSITION,  (0, 0, 1, 0)) # directional light from the front
        glLight(GL_LIGHT0, GL_POSITION,  (5, 5, 5, 1)) # point light from the left, top, front
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))

        glEnable(GL_DEPTH_TEST) 

        self.objects = [ ]

        self.default_pos = (0 ,  0 , -5)

        glTranslatef(*self.default_pos )


    def add_object(self, obj ):
        self.objects.append( obj )

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )

        for obj in self.objects:
            glPushMatrix()
            obj.draw()
            glPopMatrix()


        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)
        glDisable(GL_COLOR_MATERIAL)

        pygame.display.flip()




class Sphere(object):
    def __init__(self , x , y , z  , r , color=(1,1,1) ):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.color = color 


    def draw(self):

        glColor3f(*self.color )

        sphere = gluNewQuadric()

        
        glTranslatef(self.x ,self.y ,self.z );
        gluSphere(sphere, self.r  , 32, 32)



class Cylinder(object):
    def __init__(self , x , y , z  , r , h  , color=(1,1,1) ):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.h = h 
        self.color = color 


    def draw(self):

        glColor3f(*self.color )

        sphere = gluNewQuadric()
        glRotatef(45.0, 1.0, 1.0, 1.0)
        
        glTranslatef(self.x ,self.y ,self.z )
        

        gluCylinder(sphere, self.r  , self.r  , self.h  ,32,32)



class Cylinder2P (object):
    def __init__(self , v1 , v2 , r  , color=(1,1,1) ):
        self.v1 = np.array(v1) 
        self.v2 = np.array(v2)
        self.r = r
        self.color = color 


    def draw(self):

        glColor3f(*self.color )

        v1 = self.v1
        v2 = self.v2 
        r = self.r 
        
        v2r = v2 - v1
        z = np.array([0.0, 0.0, 1.0])
        # the rotation axis is the cross product between Z and v2r
        ax = np.cross(z, v2r)
        l = np.sqrt(np.dot(v2r, v2r))
        # get the angle using a dot product
        angle = 180.0 / np.pi * np.arccos(np.dot(z, v2r) / l)

        
        glTranslatef(v1[0], v1[1], v1[2])
        
        #print "The cylinder between %s and %s has angle %f and axis %s\n" % (v1, v2, angle, ax)
        glRotatef(angle, ax[0], ax[1], ax[2])
        # glutSolidCylinder( )

        sphere = gluNewQuadric()
        gluCylinder(sphere, r , r   , l, 20, 20 )


        

class Box(object):
    def __init__(self , x , y , z  , w , h , d  , color=(1,1,1) ):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.d = d
        self.color = color 


    def draw(self):

        glColor3f(*self.color )

        [tx , ty , tz ] = glGetFloatv(GL_MODELVIEW_MATRIX)[3][:3]
        glTranslatef(self.x ,self.y ,self.z );

        glScalef( self.w , self.h , self.d )

        
        
        glutSolidCube(  1   )


 
if __name__ == "__main__":


    s = Scene()
    sp = Cylinder2P( [0,0,0] , [1,1,1] , 0.05  )
    sp3 = Cylinder2P( [1,1,1] , [0,1,0] , 0.05  )

    sp2 = Sphere(1,1,1,0.05 )

    w = 0.3
    h = 1
    d = 1.4

    for dd in [ -d/2 , d/2 ]:
        for hh in [-h/2 , h/2 ]:
            for ww in [-w/2 , w/2 ]:
                s.add_object( Sphere(1+ww,1+hh,1+dd ,0.05 , (1,0,0)) )


    sp4 = Box(1,1,1,w , h , d  )

    s.add_object( sp )
    s.add_object( sp2 )
    s.add_object( sp3 )
    s.add_object( sp4 )

    while True:
        # sp4.h -= 0.01
        s.render()
