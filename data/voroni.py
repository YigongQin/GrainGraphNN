#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:54:01 2022

@author: yigongqin
"""

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import random
from matplotlib import cm
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
#from possion_disk import poisson_disc_samples


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


def random_color(as_str=True, alpha=1):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(np.array(rgb)/255) + [alpha]
    
    
    
    
def plot_polygons(polygons, ax=None, alpha=0.5, linewidth=0.7, pic_size=(100,100), saveas=None, show=True):
    # Configure plot 
   # if ax is None:
   #     plt.figure(figsize=(5,5))
   #     ax = plt.subplot(111)

    # Remove ticks
    #ax.set_xticks([])
   # ax.set_yticks([])

#    ax.axis("equal")

    # Set limits
  #  ax.set_xlim(0,1)
   # ax.set_ylim(0,1)

    image = Image.new("L", (pic_size[0], pic_size[1]))       
    draw = ImageDraw.Draw(image)
      
    # Add polygons 
    for poly in polygons:
        p = []
        orientation = random.randint(0,255)
        poly = np.asarray(poly*pic_size[0], dtype=int) 
        for i in range(poly.shape[0]):
            p.append(tuple(poly[i]))
        
        draw.polygon(p, fill=orientation) 
        '''
        colored_cell = Polygon(poly,
                               linewidth=linewidth, 
                               alpha=alpha,
                               #facecolor=random_color(as_str=False),
                               facecolor = cm.coolwarm(random.randint(0,255)),
                               edgecolor="black")
        
        ax.add_patch(colored_cell)
        '''

        
    '''    
    if not saveas is None:
        plt.savefig(saveas, dpi=800)
    if show:
        plt.show()
    '''
   # plot_polygons.colored_cell = np.frombuffer(plt.tostring_rgb(), dtype=np.uint8)
    
    return np.array(image)


#n = 1000
#random_seeds = np.random.rand(n, 2)
#random_seeds=poisson_disc_samples(width=1, height=1, r=0.03)

def in_bound(x, y):
    eps = 1e-12
    if x>=-eps and x<=1+eps and y>=-eps and y<=1+eps:
        return True
    else:
        return False
        

def hexagonal_lattice(dx=0.05, noise=0.0001):
    # Assemble a hexagonal lattice
    rows, cols = int(1/dx)+1, int(1/dx)
    print('cols, rows of grains', cols, rows)
    points = []
    in_points = []
    randNoise = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*noise, size=rows*cols*5)
    count = 0
    for row in range(rows*2):
        for col in range(cols):
            count+=1
            x = ( (col + (0.5 * (row % 2)))*np.sqrt(3) )*dx
            y = row*0.5 *dx
         
            x += randNoise[count,0]
            y += randNoise[count,1]
            
            if in_bound(x, y):
              in_points.append([x,y])
              points.append([x,y])
              points.append([-x,y])
              points.append([x,-y])
              points.append([2-x,y])
              points.append([x,2-y])

    return points, in_points



def inbound_voronoi():

    mirrored_seeds, seeds = hexagonal_lattice(dx=0.1, noise=0.00005)
    vor = Voronoi(mirrored_seeds)     

    regions = []
    reordered_regions = []
    vertices = []
    vert_map = {}
    vert_count = 0
    edges = []
    
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not in_bound(x,y):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
            reordered_region = []
            for index in region:
                point = tuple(vor.vertices[index])
                if point not in vert_map:
                    vertices.append(point)
                    vert_map[point] = vert_count
                    reordered_region.append(vert_count)
                    vert_count += 1
                else:
                    reordered_region.append(vert_map[point])
            for i in range(len(region)):
                cur = vor.vertices[region[i]]
                nxt = vor.vertices[region[i+1]] if i<len(region)-1 else vor.vertices[region[0]]
                edges.append([vert_map[tuple(cur)], vert_map[tuple(nxt)]])
            reordered_regions.append(reordered_region)
                
    vor.filtered_points = seeds
    vor.filtered_regions = regions
    vor.graph_regions = reordered_regions
    vor.graph_vertices = np.array(vertices)
    vor.graph_edges = edges
    
    polygon_coors = []
    for reg in vor.graph_regions:
        polygon = vor.graph_vertices[reg]
        polygon_coors.append(polygon)
    
    
    return vor, polygon_coors
    



#vor = filter_voronoi(vor, seeds)



    
    
    
    
    
    
    
#return polygons
L = 1000
pic_size = (L, L)

vor, polygon_coors = inbound_voronoi()
image = plot_polygons(polygon_coors, pic_size=pic_size, saveas='./voronoi.png')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(vor.graph_vertices[:,0],vor.graph_vertices[:,1],s=2)
ax[0].axis("equal")
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
ax[0].set_xticks([])
ax[0].set_yticks([])

for x, y in vor.graph_edges:
    s = [vor.graph_vertices[x][0], vor.graph_vertices[y][0]]
    t = [vor.graph_vertices[x][1], vor.graph_vertices[y][1]]
    ax[1].plot(s, t, 'k')
ax[1].axis("equal")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].imshow(image, origin='lower', cmap='coolwarm')
ax[2].set_xticks([])
ax[2].set_yticks([])

plt.savefig('./voronoi.png', dpi=400)













