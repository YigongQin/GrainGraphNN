#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:27:17 2022

@author: yigongqin
"""

from math import pi, cos, sin, sqrt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class vertex:
    def __init__(self, idx, x, y, adjVertex=None):
        self.idx = idx
        self.x = x
        self.y = y
        self.neighbors = adjVertex
    
    def updateLoc(self, dx,dy):
        self.x += dx
        self.y += dy
        
    def drawVert(self, board):
        board[self.x,self.y]= np.NaN
        
        

class polygon:
    def __init__(self, alpha, vertices_list=None):
        self.alpha = alpha
        self.vertices = vertices_list
        self.image = None
    def initHexCVT(self, center, size):
        for i in range(6):
            theta = i*pi/3
            self.vertices.append((center[0]+size*cos(theta), center[1]+size*sin(theta)))
        
    def drawPoly(self, pic):
        
        image = Image.new("L", (pic.size[0], pic.size[1]))       
        draw = ImageDraw.Draw(image)
        draw.polygon(self.vertices, fill=self.alpha)   
        self.image = image
        pic.board[np.array(image)>0] = self.alpha

        
class graph:
    def __init__(self, size, defaultNumVert = 100):
     #   self.sizeX = sizeX
     #   self.sizeY = sizeY
        self.boardsize = size
        self.board = np.zeros(size)
        self.vertices = [] 
        self.edges = []
        self.numVert = defaultNumVert
        self.A = None
        self.vertices.append(vertex(0, 0, 0))
        self.vertices.append(vertex(1, 0, size[1]-1))
        self.vertices.append(vertex(2, size[0]-1, 0))
        self.vertices.append(vertex(3, size[0]-1, size[1]-1))
        self.
    
    def randVert(self):
        for i in range(4, self.numVert):
            start = int(np.random.random()*self.boardsize[0])
            end = int(np.random.random()*self.boardsize[1])
            self.vertices.append(vertex(i, start, end))
    
    def CVTvert(self):
        
        self.vertices.append()
        
    def showVert(self):
        fig,ax = plt.subplots()
        for vert in self.vertices:
            vert.drawVert(self.board)
        ax.imshow(self.board)
        
    def VoronoiCell(self):
        vor = Voronoi(points)
        fig = voronoi_plot_2d(vor)
        plt.show()
        
    def add_patch(self):
        
        a = polygon(np.random.random(), )
        a.drawPoly(self.board)
        
    def image2vert(self):
        
        return self.vertices
    
    def create_edges(self):
        
        A = None
        




initGrainSize = 50
size = (201,201)
g1 = graph(size)        
#a = polygon(alpha=100)
#a.initHexCVT([0,0], initGrainSize)
#a.drawPoly(board)







###############

## the plotting path: give vertices coordinates, form polygon by anticlockwise edge link, insert angle

#############



'''

for i in range(0, size[0], 3*initGrainSize):
    for j in range(0, size[1], int(sqrt(3)*initGrainSize)):
        print(i,j)
        CVT = polygon(alpha = 100)
        CVT.initHexCVT([i,j], initGrainSize)
        CVT.drawPoly(board)



image = Image.new("RGB", (640, 480))

draw = ImageDraw.Draw(image)

# points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
draw.polygon((points), fill=(100,100,100))

image.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

y = np.array([[1,1], [2,1], [2,2], [1,2], [0.5,1.5]])

p = Polygon(y, facecolor = 'k')

fig,ax = plt.subplots()

ax.add_patch(p)
ax.set_xlim([0,3])
ax.set_ylim([0,3])
plt.show()

'''
