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
from possion_disk import poisson_disc_samples

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)




def random_color(as_str=True, alpha=0.5):
    rgb = [random.randint(0,255),
           random.randint(0,255),
           random.randint(0,255)]
    if as_str:
        return "rgba"+str(tuple(rgb+[alpha]))
    else:
        # Normalize & listify
        return list(np.array(rgb)/255) + [alpha]
    
    
    
    
def plot_polygons(polygons, ax=None, alpha=0.5, linewidth=0.7, saveas=None, show=True):
    # Configure plot 
    if ax is None:
        plt.figure(figsize=(5,5))
        ax = plt.subplot(111)

    # Remove ticks
    #ax.set_xticks([])
   # ax.set_yticks([])

    ax.axis("equal")

    # Set limits
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    # Add polygons 
    for poly in polygons:
        colored_cell = Polygon(poly,
                               linewidth=linewidth, 
                               alpha=alpha,
                               facecolor=random_color(as_str=False, alpha=1),
                               edgecolor="black")
        ax.add_patch(colored_cell)

    if not saveas is None:
        plt.savefig(saveas, dpi=800)
    if show:
        plt.show()

    return ax 


n = 1000
#random_seeds = np.random.rand(n, 2)
random_seeds=poisson_disc_samples(width=1, height=1, r=0.03)


def hexagonal_lattice(rows=20, cols=20, noise=0.0001):
    # Assemble a hexagonal lattice
    points = []
    for row in range(rows*2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2)))*np.sqrt(3)
            y = row*0.5
            points.append((x/rows, y/cols))
    points = np.asarray(points)
    randNoise = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)*noise, size=points.shape[0])
    
    points += randNoise

    return points

random_seeds = hexagonal_lattice()
vor = Voronoi(random_seeds) 
regions, vertices = voronoi_finite_polygons_2d(vor)


polygons = []
for reg in regions:
    polygon = vertices[reg]
    polygons.append(polygon)
#return polygons

plot_polygons(polygons, saveas='./voronoi.png')



