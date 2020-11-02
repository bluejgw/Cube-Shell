# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:16:58 2020

@author: bluejgw
"""
import pyvista as pv
import sympy as sp
from sympy import Matrix, lambdify
import numpy as np
from PyQt5 import Qt, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from pyvistaqt import QtInteractor
import sys, os
#import meshio
import trimesh

# from CGAL import CGAL_Polygon_mesh_processing
# current conda cgal is version 5.0.1, it doesn't include centroid()
# either wait till 5.0.3 is released on conda or DIY

# initiate stored mesh
mesh = pv.PolyData()

class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        editMenu = mainMenu.addMenu('Edit')
        
        # opening a mesh file
        self.open_mesh_action = Qt.QAction('Open Mesh...', self)
        self.open_mesh_action.triggered.connect(self.open_mesh)
        fileMenu.addAction(self.open_mesh_action)
        
        # exit button
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # inserting maximally inscribed cube via cone intersection
        self.max_cube_cone_action = Qt.QAction('Max Cube - Cone', self)
        self.max_cube_cone_action.triggered.connect(self.max_cube_cone)
        editMenu.addAction(self.max_cube_cone_action)

        # inserting maximally inscribed cube via ray tracing
        self.max_cube_ray_action = Qt.QAction('Max Cube - Ray', self)
        self.max_cube_ray_action.triggered.connect(self.max_cube_ray)
        editMenu.addAction(self.max_cube_ray_action)
        
        # inserting maximally inscribed cube via ray tracing
        self.ext_max_cube_action = Qt.QAction('Extend Max Cube', self)
        self.ext_max_cube_action.triggered.connect(self.ext_max_cube)
        editMenu.addAction(self.ext_max_cube_action)

        #Create Cone in Mesh
        self.next_cubes_action = Qt.QAction('Next Cubes', self)
        self.next_cubes_action.triggered.connect(self.next_cubes)
        editMenu.addAction(self.next_cubes_action)
        
        # slice mesh horizontally based on internal cubes
        self.cube_hslice_action = Qt.QAction('Cube H-Slice', self)
        self.cube_hslice_action.triggered.connect(self.cube_hslice)
        editMenu.addAction(self.cube_hslice_action)

        # slice mesh (interactively)
        self.slice_action = Qt.QAction('Slice', self)
        self.slice_action.triggered.connect(self.slice)
        editMenu.addAction(self.slice_action)
        
        # slice mesh with clipping (interactively)
        self.clip_slice_action = Qt.QAction('Clip Slice', self)
        self.clip_slice_action.triggered.connect(self.clip_slice)
        editMenu.addAction(self.clip_slice_action)

        # create bounding box(es) for mesh (interactively)
        self.bounding_action = Qt.QAction('Bounding', self)
        self.bounding_action.triggered.connect(self.bounding_bar)
        editMenu.addAction(self.bounding_action)
        
        if show:
            self.show()

        self.plotter.add_axes(interactive=None, line_width=2, color=None, x_color=None, y_color=None, z_color=None, xlabel='X', ylabel='Y', zlabel='Z', labels_off=False, box=None, box_args=None)

    def open_mesh(self):
        """ add a mesh to the pyqt frame """
        global mesh

        # open file
        file_info = QtWidgets.QFileDialog.getOpenFileName()
        file_dir = file_info[0]
        
        # determine file type and if conversion needed
        file_name = os.path.split(file_dir)
        file_name_part = os.path.splitext(file_name[1])

        # convert mesh file type
        #if ext != ".vtk" or ext != ".VTK":
        #    mesh = meshio.read(file_dir)
        #    meshio.write(root + ".vtk", mesh)
        #    mesh = pv.read(head + "/" + root + ".vtk")
            # need to store elsewhere or delete .vtk file in the future
        #else:
        #    mesh = pv.read(file_dir)

        # read mesh & transform according to principal axes
        pre = trimesh.load_mesh(file_dir)
        orient = pre.principal_inertia_transform
        pre = pre.apply_transform(orient)
        pre.export('data/'+ file_name_part[0] + '_oriented.STL')
        mesh = pv.read('data/'+ file_name_part[0] + '_oriented.STL')

        # show transformed mesh
        self.plotter.add_mesh(mesh, show_edges=True, color="w", opacity=0.6)

        # reset plotter
        self.reset_plotter()

        # find mesh centroid and translate the mesh so that's the origin
        #self.centroid()

        # show bounding box
        self.plotter.add_bounding_box(opacity=0.5, color="y")

    def reset_plotter(self):
        """ clear plotter of mesh or interactive options """
        # clear plotter
        self.plotter.clear()
        self.plotter.clear_plane_widgets()
        self.plotter.reset_camera()
        #self.update()
        
        # callback opened mesh
        self.plotter.add_mesh(mesh, show_edges=True, color="w", opacity=0.6)
        
        # show origin
        self.plotter.add_axes_at_origin(xlabel='X', ylabel='Y', zlabel='Z', line_width=3, labels_off=False)

        # show floors
        #self.plotter.add_floor('-y')
        #self.plotter.add_floor('-z')
        
    # def centroid(self):
    #     """ find centroid volumetrically and indicate on graph """
    #     global Vol_centroid, V

    #     # find the vertices & the vertex indices of each triangular face
    #     V = np.array(mesh.points)
    #     col = len(V)
    #     f_ind = np.array(mesh.faces.reshape((-1,4))[:, 1:4])
        
    #     # define an arbitrary start point from middle of max and min of X,Y,Z of
    #     # all points: in a convex manifold it falls inside the volume (requires
    #     # segmentation for general application)
    #     start = np.array(mesh.center)
    #     X_start = start[0]
    #     Y_start = start[1]
    #     Z_start = start[2]
        
    #     # initialize variables
    #     centroids = []
    #     Vol_total = 0
    #     Sum_vol_x = 0
    #     Sum_vol_y = 0
    #     Sum_vol_z = 0
        
    #     # find centroid from all tetrahedra made with arbitrary center and triangular faces
    #     for i in range(0, col-1, 3):          
    #         # find the center of each tetrahedron (average of X,Y,Z of 
    #         # 4 vertices, 3 from the triangle, and one arbitrary start point)
    #         X_cent = (X_start + V[f_ind[i,0],0] + V[f_ind[i+1,0],0] + V[f_ind[i+2,0],0]) / 4
    #         Y_cent = (Y_start + V[f_ind[i,1],1] + V[f_ind[i+1,1],1] + V[f_ind[i+2,1],1]) / 4
    #         Z_cent = (Z_start + V[f_ind[i,2],2] + V[f_ind[i+1,2],2] + V[f_ind[i+2,2],2]) / 4
    
    #         # compute the volume of each tetrahedron
    #         V1 = np.array([V[f_ind[i,0],0], V[f_ind[i,1],1], V[f_ind[i,2],2]])**2 - np.array([X_start, Y_start, Z_start])**2
    #         V2 = np.array([V[f_ind[i+1,0],0], V[f_ind[i+1,1],1], V[f_ind[i+1,2],2]])**2 - np.array([V[f_ind[i,0],0], V[f_ind[i,1],1], V[f_ind[i,2],2]])**2
    #         V3 = np.array([V[f_ind[i+2,0],0], V[f_ind[i+2,1],1], V[f_ind[i+2,2],2]])**2 - np.array([V[f_ind[i+1,0],0], V[f_ind[i+1,1],1], V[f_ind[i+1,2],2]])**2
    #         V1 = V1.reshape((-1,1))
    #         V2 = V2.reshape((-1,1))
    #         V3 = V3.reshape((-1,1))
    #         Vol = abs(np.linalg.det(np.hstack([V1, V2, V3]))) / 6
    
    #         # tally up each cycle
    #         Vol_total = Vol_total + Vol
    #         Sum_vol_x = Sum_vol_x + Vol * X_cent
    #         Sum_vol_y = Sum_vol_y + Vol * Y_cent
    #         Sum_vol_z = Sum_vol_z + Vol * Z_cent
    #         centroids.append([X_cent,Y_cent,Z_cent])
        
    #     # find & show centroid
    #     centroids = np.asarray(centroids)
    #     Vol_centroid = [Sum_vol_x, Sum_vol_y, Sum_vol_z] / Vol_total
    #     print("Total Volume:", Vol_total)
    #     print("Centroid:", Vol_centroid)

    def max_cube_cone(self):
        """ add a maximally inscribed cube within the opened mesh (via cone intersection) """
        global ranges, max_c1, max_c2, nearest_vert
        global face_center, max_cube, max_normal
        global cube_V, cube_F
        global Vol_centroid, V

        # find the vertices & the vertex indices of each triangular face
        V = np.array(mesh.points)

        # reset plotter
        self.reset_plotter()

        # show centroid
        Vol_centroid = np.array([0,0,0])
        self.plotter.add_mesh(pv.PolyData(Vol_centroid), color='r', point_size=20.0, render_points_as_spheres=True)

        # project cones to from centroid to find maximally inscribed cube
        ranges = mesh.bounds
        h = (abs(ranges[4]) + abs(ranges[5]))/3
        ang = np.arctan(0.5/(np.sqrt(2)/2))
        ang = float(90 - np.degrees(ang))
        max_c1 = pv.Cone(center=Vol_centroid+[0,0,h/2], direction=[0,0,-1], height=h, radius=None, capping=False, angle=ang, resolution=100)
        max_c2 = pv.Cone(center=Vol_centroid-[0,0,h/2], direction=[0,0,1], height=h, radius=None, capping=False, angle=ang, resolution=100)
        self.plotter.add_mesh(max_c1, color="r", show_edges=True, opacity=0.4)
        self.plotter.add_mesh(max_c2, color="r", show_edges=True, opacity=0.4)
        
        # find the nearest possible cube vertex from top cone & mesh intersection
        top_clip = mesh.clip_surface(max_c1, invert=True)
        top_vert = np.array(top_clip.points)
        top = self.nearest_pt(top_vert, Vol_centroid)

        # find the nearest possible cube vertex from bottom cone & mesh intersection
        bottom_clip = mesh.clip_surface(max_c2, invert=True)
        bottom_vert = np.array(bottom_clip.points)
        bottom = self.nearest_pt(bottom_vert, Vol_centroid)

        # show top & bottom clipped surfaces
        #self.plotter.add_mesh(top_clip, opacity=0.6, show_edges=True, color="g")
        #self.plotter.add_mesh(bottom_clip, opacity=0.6, show_edges=True, color="g")

        # find the nearest possible cube vertex between the two
        if top[0] < bottom[0]:
            p = top[1]
            V = top[2]
        else:
            p = bottom[1]
            V = bottom[2]
        
        # create max cube from nearest possible cube vertex
        cube_V, cube_F = self.create_cube(V[p,:], Vol_centroid, 'z')
        max_cube = pv.PolyData(cube_V, cube_F)

        # show max cube
        self.plotter.add_mesh(max_cube, show_edges=True, color="b", opacity=0.6)

        # record nearest vertex
        nearest_vert = V[p,:]

        # find & show max cube face centers
        cell_center = max_cube.cell_centers()
        face_center = np.array(cell_center.points)
        self.plotter.add_mesh(cell_center, color="r", point_size=8.0, render_points_as_spheres=True)

        # find max cube face normals
        max_normal = max_cube.cell_normals

        #rot_z = np.dot(np.array(max_normal[1]), np.array([1,0,0]))/ np.linalg.norm(np.array(max_normal[1])) / np.linalg.norm(np.array([1,0,0]))
        #print(rot_z)

        #mesh = mesh.rotate_z(rot_z)
        #max_cube = max_cube.rotate_z(rot_z)
        #self.plotter.remove_actor(mesh)
        #self.plotter.add_mesh(mesh, show_edges=True, color="w", opacity=0.6)
        #self.plotter.add_mesh(max_cube, show_edges=True, color="b", opacity=0.6)

    def max_cube_ray(self):
        """ add a maximally inscribed cube within the opened mesh (via ray tracing) """
        global ranges, nearest_vert
        global face_center, max_cube, max_normal
        global cube_V, cube_F
        global Vol_centroid, V

        # find the vertices & the vertex indices of each triangular face
        V = np.array(mesh.points)

        # reset plotter
        self.reset_plotter()

        # show centroid
        Vol_centroid = np.array([0,0,0])
        self.plotter.add_mesh(pv.PolyData(Vol_centroid), color='r', point_size=20.0, render_points_as_spheres=True)

        # project rays from centroid to find maximally inscribed cube
        ranges = mesh.bounds

        # find the nearest possible cube vertex from top rays & mesh intersection
        top_vert = self.vert_ray(Vol_centroid, 'z')
        top = self.nearest_pt(top_vert, Vol_centroid)

        # find the nearest possible cube vertex from bottom rays & mesh intersection
        bottom_vert = self.vert_ray(Vol_centroid, '-z')
        bottom = self.nearest_pt(bottom_vert, Vol_centroid)        

        # find the nearest possible cube vertex between the two
        if top[0] < bottom[0]:
            p = top[1]
            V = top[2]
        else:
            p = bottom[1]
            V = bottom[2]
        
        # create and show max cube
        #cube_V, cube_F = self.create_cube(V[p,:], Vol_centroid, 'z')
        cube_V, cube_F = self.create_cube(V[p,:], Vol_centroid, [0,0,1])
        max_cube = pv.PolyData(cube_V, cube_F)
        self.plotter.add_mesh(max_cube, show_edges=True, color="b", opacity=0.6)
        print(max_cube.volume)

        # record nearest vertex
        nearest_vert = V[p,:]

        # find & show max cube face centers
        cell_center = max_cube.cell_centers()
        face_center = np.array(cell_center.points)
        self.plotter.add_mesh(cell_center, color="r", point_size=8.0, render_points_as_spheres=True)

        # find max cube face normals
        max_normal = max_cube.cell_normals

    def vert_ray(self, start, dir):
        ''' from starting point shoot out 8 rays to find vertices of a possible cube,
        whose face normals would be in either in x,y,z direction, or rotated 45 deg along z-axis '''
        # set ray directions
        if dir == 'z':
            r1_dir = np.array([1,1,1])
            r2_dir = np.array([1,0,1])
            r3_dir = np.array([1,-1,1])
            r4_dir = np.array([0,-1,1])
            r5_dir = np.array([-1,-1,1])
            r6_dir = np.array([-1,0,1])
            r7_dir = np.array([-1,1,1])
            r8_dir = np.array([0,1,1])
        elif dir == '-z':
            r1_dir = np.array([1,1,-1])
            r2_dir = np.array([1,0,-1])
            r3_dir = np.array([1,-1,-1])
            r4_dir = np.array([0,-1,-1])
            r5_dir = np.array([-1,-1,-1])
            r6_dir = np.array([-1,0,-1])
            r7_dir = np.array([-1,1,-1])
            r8_dir = np.array([0,1,-1])

        # set ray length
        r_len = abs(ranges[4] - ranges[5])/2 * np.sqrt(1**2 + (np.sqrt(2)/2)**2)

        # set ray end points
        r1_end = start + r1_dir / np.linalg.norm(r1_dir) * r_len
        r2_end = start + r2_dir / np.linalg.norm(r2_dir) * r_len
        r3_end = start + r3_dir / np.linalg.norm(r3_dir) * r_len
        r4_end = start + r4_dir / np.linalg.norm(r4_dir) * r_len
        r5_end = start + r5_dir / np.linalg.norm(r5_dir) * r_len
        r6_end = start + r6_dir / np.linalg.norm(r6_dir) * r_len
        r7_end = start + r7_dir / np.linalg.norm(r7_dir) * r_len
        r8_end = start + r8_dir / np.linalg.norm(r8_dir) * r_len
        
        # perform ray trace
        r1_pts, r1_ind = mesh.ray_trace(start, r1_end)
        r2_pts, r2_ind = mesh.ray_trace(start, r2_end)
        r3_pts, r3_ind = mesh.ray_trace(start, r3_end)
        r4_pts, r4_ind = mesh.ray_trace(start, r4_end)
        r5_pts, r5_ind = mesh.ray_trace(start, r5_end)
        r6_pts, r6_nd = mesh.ray_trace(start, r6_end)
        r7_pts, r7_ind = mesh.ray_trace(start, r7_end)
        r8_pts, r8_ind = mesh.ray_trace(start, r8_end)

        # initialize rays
        r1 = pv.Line(start, r1_end)
        r2 = pv.Line(start, r2_end)
        r3 = pv.Line(start, r3_end)
        r4 = pv.Line(start, r4_end)
        r5 = pv.Line(start, r5_end)
        r6 = pv.Line(start, r6_end)
        r7 = pv.Line(start, r7_end)
        r8 = pv.Line(start, r8_end)

        # initialize intersections
        r1_int = pv.PolyData(r1_pts[0])
        r2_int = pv.PolyData(r2_pts[0])
        r3_int = pv.PolyData(r3_pts[0])
        r4_int = pv.PolyData(r4_pts[0])
        r5_int = pv.PolyData(r5_pts[0])
        r6_int = pv.PolyData(r6_pts[0])
        r7_int = pv.PolyData(r7_pts[0])
        r8_int = pv.PolyData(r8_pts[0])

        # show rays
        l_wid = 5
        self.plotter.add_mesh(r1, color='g', line_width=l_wid)
        self.plotter.add_mesh(r2, color='g', line_width=l_wid)
        self.plotter.add_mesh(r3, color='g', line_width=l_wid)
        self.plotter.add_mesh(r4, color='g', line_width=l_wid)
        self.plotter.add_mesh(r5, color='g', line_width=l_wid)
        self.plotter.add_mesh(r6, color='g', line_width=l_wid)
        self.plotter.add_mesh(r7, color='g', line_width=l_wid)
        self.plotter.add_mesh(r8, color='g', line_width=l_wid)

        # show intersections
        pt_size = 20
        self.plotter.add_mesh(r1_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r2_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r3_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r4_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r5_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r6_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r7_int, color='g', point_size=pt_size)
        self.plotter.add_mesh(r8_int, color='g', point_size=pt_size)

        # array of intersections (assume 1 intersection per ray)
        r_int = np.vstack([r1_int.points, r2_int.points, r3_int.points, r4_int.points,
                    r5_int.points, r6_int.points, r7_int.points, r8_int.points])

        return r_int

    def nearest_pt(self, vert, starting_pt):
        """ find nearest vertex: for segmented convex manifold, a cube with volume centroid as 
        center and nearest vertex as cube vertex, it falls inside the volume """
        # find nearest point from the list of points
        c = len(vert)
        dist = np.zeros(c)
        for i in range(0, c):
            dist[i] = np.sqrt((vert[i,0] - starting_pt[0])**2 + (vert[i,1] - starting_pt[1])**2
                            + (vert[i,2] - starting_pt[2])**2)
                
        # find index of the nearest point
        nearest = min(dist)
        p = np.where(dist == nearest)
        p = p[0].item()

        return nearest, p, vert
            
    def create_cube(self, vertex, starting_pt, axis):
        ''' create cube from the nearest pt & centroid '''
        # find the other 7 vertices
        # 3 vertices can be found by rotating the first point 90 degrees 3 times around Z axis of centroid
        # 4 vertices can be found by translating the first four vertices twice the half edge
        # found from the distance times sin(pi/4)
        t = sp.Symbol('t')

        R_t = Matrix([[sp.cos(t)+axis[0]**2*(1-sp.cos(t)), axis[0]*axis[1]*(1-sp.cos(t))-axis[2]*sp.sin(t), axis[0]*axis[2]*(1-sp.cos(t))+axis[1]*sp.sin(t)],
            [axis[1]*axis[0]*(1-sp.cos(t))+axis[2]*sp.sin(t), sp.cos(t)+axis[1]**2*(1-sp.cos(t)), axis[1]*axis[2]*(1-sp.cos(t))-axis[0]*sp.sin(t)],
            [axis[2]*axis[0]*(1-sp.cos(t))-axis[1]*sp.sin(t), axis[2]*axis[1]*(1-sp.cos(t))+axis[0]*sp.sin(t), sp.cos(t)+axis[2]**2*(1-sp.cos(t))]])
        R = lambdify(t, R_t)
        #translate = np.array([0, starting_pt[1], starting_pt[2]])

        # if dir == 'x':
        #     R_t = Matrix([[0, 0, 1],
        #         [sp.cos(t), -sp.sin(t), 0],
        #         [sp.sin(t), sp.cos(t), 0]])
        #     R = lambdify(t, R_t)
        #     translate = np.array([0, starting_pt[1], starting_pt[2]])
        # elif dir == 'y':
        #     R_t = Matrix([[sp.sin(t), sp.cos(t), 0],
        #         [0, 0, 1],
        #         [sp.cos(t), -sp.sin(t), 0]])
        #     R = lambdify(t, R_t)
        #     translate = np.array([starting_pt[0], 0, starting_pt[2]])
        # elif dir == 'z':
        #     R_t = Matrix([[sp.cos(t), -sp.sin(t), 0],
        #         [sp.sin(t), sp.cos(t), 0],
        #         [0, 0, 1]])
        #     R = lambdify(t, R_t)
        #     translate = np.array([starting_pt[0], starting_pt[1], 0])
        
        # construct the array of the first 4 vertices
        # V_a = np.array(vertex-translate)
        # a_2 = np.dot(R(np.pi/2), V_a.T).T + translate
        # a_3 = np.dot(R(np.pi), V_a.T).T + translate
        # a_4 = np.dot(R(3*np.pi/2), V_a.T).T + translate
        V_a = np.array(vertex)
        a_2 = np.dot(R(np.pi/2), V_a.T).T
        a_3 = np.dot(R(np.pi), V_a.T).T
        a_4 = np.dot(R(3*np.pi/2), V_a.T).T
        self.plotter.add_mesh(pv.PolyData(V_a), color='y', point_size=10.0, render_points_as_spheres=True)
        cube_V_start = np.array([V_a, a_2, a_3, a_4])
        cube_V_start_center = np.array(pv.PolyData(cube_V_start).center)
        
        # find the translation distance
        #half_edge = np.ones((4,1)) * [[0, 0, np.sign(starting_pt[2]-vertex[2])]] * np.sqrt((vertex[0]-starting_pt[0])**2 + (vertex[1]-starting_pt[1])**2) * sp.sin(sp.pi/4)
        trans_dis = Vol_centroid - cube_V_start_center
        trans_dir = trans_dis / np.linalg.norm(trans_dis)
        half_edge = np.ones((4,1)) * [trans_dir] * np.sqrt((vertex[0]-starting_pt[0])**2 + (vertex[1]-starting_pt[1])**2) * sp.sin(sp.pi/4)
        cube_trans = np.asarray(2*half_edge, dtype=np.float64)

        # construct the cube
        cube_V_end = np.add(cube_V_start, cube_trans)
        cube_V = np.vstack((cube_V_start, cube_V_end))

        # construct the 6 faces of the cube
        cube_F = np.hstack([[4,0,1,2,3],
                        [4,0,3,7,4],
                        [4,0,1,5,4],
                        [4,1,2,6,5],
                        [4,2,3,7,6],
                        [4,4,5,6,7]])

        # test_V = cube_V_start
        # test_F = np.array([4,0,1,2,3])
        
        return cube_V, cube_F
    
    def ext_max_cube(self):
        ''' extend max cube into maximally inscribed cuboid '''
        global face_center, ext_max_cube, max_normal

        # find the 3 out of 6 normal directions the max cube can be extended towards
        ext_dir = np.empty(shape=(3,3)) 
        main_dir = Vol_centroid - nearest_vert
        ind = 0
        for i in range(0, 6):
            if np.dot(max_normal[i,:], main_dir) > 0:
                ext_dir[ind,:] = max_normal[i,:]
                ind +=1

        # extend faces by shooting a ray from the 4 vertices on each extendable face
        # in the direction of its face normal. Find the nearest intersection and
        # it would be the limit of extension for that face
        for i in range(0, 3):
            F_ind = np.where((max_normal == ext_dir[i]).all(axis=1))
            np.reshape(cube_F, (6,5))
            faces = np.reshape(cube_F, (6,5))
            V_ind = faces[F_ind][0,1:5]
            current_V = np.vstack([cube_V[V_ind[0]], cube_V[V_ind[1]], cube_V[V_ind[2]], cube_V[V_ind[3]]])
            ext_V = self.ext_ray(current_V, ext_dir[i])
            cube_V[V_ind] = ext_V

        # create & show extended max cube
        ext_max_cube = pv.PolyData(cube_V, cube_F)
        self.plotter.add_mesh(ext_max_cube, show_edges=True, color="y", opacity=0.6)

        # find face centers of extended max cube
        cell_center = ext_max_cube.cell_centers()
        face_center = np.array(cell_center.points)

        # find face normals of the extended max cube
        max_normal = ext_max_cube.cell_normals

    def ext_ray(self, current_V, ext_dir):
        ''' shoot rays from vertices of a cube face towards face normal & obtain intersections with mesh '''
        # initialize variables
        ext_end = current_V + ext_dir * np.ones((4,1))
        ext_int = [None] * 4
        ext_dis = np.zeros(4)

        # perform ray tracing per extending face vertex
        for i in range(0,4):
            ext_int, ind = mesh.ray_trace(current_V[i], ext_end[i])
            ext_dis[i] = np.sqrt((ext_int[0][0] - current_V[i][0])**2 + (ext_int[0][1] - current_V[i][1])**2
                                 + (ext_int[0][2] - current_V[i][2])**2)

        # extend vertices by the shortest intersection distance
        ext_V = current_V + ext_dir * np.ones((4,1)) * min(ext_dis)
        
        return ext_V

    def next_cubes(self):
        ''' create cubes within the mesh from the face centers of the first cube'''
        # project 6 cones from max cube face centers
        ranges = mesh.bounds
        h_x = (abs(ranges[0]) + abs(ranges[1]))/3
        h_y = (abs(ranges[2]) + abs(ranges[3]))/3
        h_z = (abs(ranges[4]) + abs(ranges[5]))/3
        ang = np.arctan(1/(np.sqrt(2)/2))
        ang = float(90 - np.degrees(ang))

        # from the faces of max cube initialize 6 cones, whose surface
        # represents possible locations of next cube vertices
        x_c1 = pv.Cone(center=face_center[1] + h_x/2 * max_normal[1], direction = -max_normal[1], height = h_x, radius=None, resolution= 100, angle = ang, capping=False)
        x_c2 = pv.Cone(center=face_center[3] + h_x/2 * max_normal[3], direction = -max_normal[3], height = h_x, radius=None, resolution= 100, angle = ang, capping=False)
        y_c1 = pv.Cone(center=face_center[2] + h_y/2 * max_normal[2], direction = -max_normal[2], height = h_y, radius=None, resolution= 100, angle = ang, capping=False)
        y_c2 = pv.Cone(center=face_center[4] + h_y/2 * max_normal[4], direction = -max_normal[4], height = h_y, radius=None, resolution= 100, angle = ang, capping=False)
        z_c1 = pv.Cone(center=face_center[0] + h_z/2 * max_normal[0], direction = -max_normal[0], height = h_z, radius=None, resolution= 100, angle = ang, capping=False)
        z_c2 = pv.Cone(center=face_center[5] + h_z/2 * max_normal[5], direction = -max_normal[5], height = h_z, radius=None, resolution= 100, angle = ang, capping=False)
        
        # show cones
        self.plotter.add_mesh(x_c1,color="r", opacity=0.2, show_edges=True)
        # self.plotter.add_mesh(x_c2,color="r", opacity=0.2, show_edges=True)
        # self.plotter.add_mesh(y_c1,color="y", opacity=0.2, show_edges=True)
        # self.plotter.add_mesh(y_c2,color="y", opacity=0.2, show_edges=True)
        # self.plotter.add_mesh(z_c1,color="g", opacity=0.2, show_edges=True)
        # self.plotter.add_mesh(z_c2,color="g", opacity=0.2, show_edges=True)

        # clip mesh with all cones from max cube face centers
        x1_clip = mesh.clip_surface(x_c1, invert=True)
        x2_clip = mesh.clip_surface(x_c2, invert=True)
        y1_clip = mesh.clip_surface(y_c1, invert=True)
        y2_clip = mesh.clip_surface(y_c2, invert=True)
        z1_clip = mesh.clip_surface(z_c1, invert=True)
        z2_clip = mesh.clip_surface(z_c2, invert=True)

        self.plotter.add_mesh(x1_clip, color="y", opacity=1)

        # find vertices in meshes cipped by cones
        x1_vert = np.array(x1_clip.points)
        x2_vert = np.array(x2_clip.points)
        y1_vert = np.array(y1_clip.points)
        y2_vert = np.array(y2_clip.points)
        z1_vert = np.array(z1_clip.points)
        z2_vert = np.array(z2_clip.points)

        # find nearest vertices in meshes clipped by cones
        x1 = self.nearest_pt(x1_vert, face_center[1])
        x2 = self.nearest_pt(x2_vert, face_center[3])
        y1 = self.nearest_pt(y1_vert, face_center[2])
        y2 = self.nearest_pt(y2_vert, face_center[4])
        z1 = self.nearest_pt(z1_vert, face_center[0])
        z2 = self.nearest_pt(z2_vert, face_center[5])

        # create cubes from nearest vertices
        x_c1_V, x_c1_F = self.create_cube(x1[2][x1[1],:], face_center[1], max_normal[1])
        x_c2_V, x_c2_F = self.create_cube(x2[2][x2[1],:], face_center[3], max_normal[3])
        y_c1_V, y_c1_F = self.create_cube(y1[2][y1[1],:], face_center[2], max_normal[2])
        y_c2_V, y_c2_F = self.create_cube(y2[2][y2[1],:], face_center[4], max_normal[4])
        z_c1_V, z_c1_F = self.create_cube(z1[2][z1[1],:], face_center[0], max_normal[0])
        z_c2_V, z_c2_F = self.create_cube(z2[2][z2[1],:], face_center[5], max_normal[5])

        # test_V = x_c2_V[0:4,:]
        # test_F = np.array([4,0,1,2,3])
        # x_cube2 = pv.PolyData(test_V, test_F)

        x_cube1 = pv.PolyData(x_c1_V, x_c1_F)
        x_cube2 = pv.PolyData(x_c2_V, x_c2_F)
        y_cube1 = pv.PolyData(y_c1_V, y_c1_F)
        y_cube2 = pv.PolyData(y_c2_V, y_c2_F)
        z_cube1 = pv.PolyData(z_c1_V, z_c1_F)
        z_cube2 = pv.PolyData(z_c2_V, z_c2_F)

        # show next cubes
        self.plotter.add_mesh(x_cube1, show_edges=True, color="b", opacity=0.6)
        # self.plotter.add_mesh(x_cube2, show_edges=True, color="b", opacity=0.6)
        # self.plotter.add_mesh(y_cube1, show_edges=True, color="b", opacity=0.6)
        # self.plotter.add_mesh(y_cube2, show_edges=True, color="b", opacity=0.6)
        # self.plotter.add_mesh(z_cube1, show_edges=True, color="b", opacity=0.6)
        # self.plotter.add_mesh(z_cube2, show_edges=True, color="b", opacity=0.6)

    def cube_hslice(self):
        """ slice mesh horizontally based on internal cubes """
        # reset plotter
        self.reset_plotter()
        
        # create sliced parts
        part1 = mesh.clip_closed_surface('z', origin=face_center[0])
        part2_a = mesh.clip_closed_surface('-z', origin=face_center[0])
        part2 = part2_a.clip_closed_surface('z', origin=face_center[5])
        part3 = mesh.clip_closed_surface('-z', origin=face_center[5])

        # display sliced parts
        self.plotter.clear()
        self.plotter.add_mesh(max_cube, show_edges=True, color="b", opacity=0.6)
        self.plotter.add_mesh(pv.PolyData(Vol_centroid), color='r', point_size=20.0, render_points_as_spheres=True)
        self.plotter.add_mesh(part1, show_edges=True, color="r", opacity=0.4)
        self.plotter.add_mesh(part2, show_edges=True, color="w", opacity=0.4)
        self.plotter.add_mesh(part3, show_edges=True, color="g", opacity=0.4)

    def slice(self):
        """ slice the mesh interactively """
        # reset plotter
        self.reset_plotter()

        self.plotter.add_mesh_slice_orthogonal(mesh)
    
    def clip_slice(self):
        """ slice & clip the mesh interactively """     
        # reset plotter
        self.reset_plotter()

        self.plotter.add_mesh_clip_plane(mesh)

    def bounding(self, level):
        level = int(level)
        bound = mesh.obbTree
        bound.SetMaxLevel(10)
        bound.GenerateRepresentation(level, boxes)
        self.plotter.add_mesh(boxes, opacity=0.2, color="g")
        return

    def bounding_bar(self):
        """ show various levels of OBB (Oriented Bounding Box) interactively """  
        # initialize bounding boxes mesh
        global boxes
        boxes = pv.PolyData()

        # reset plotter
        self.reset_plotter()

        self.plotter.add_slider_widget(self.bounding, [0, 10], title='Level')

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Window Close", "Are you sure you want to quit program?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        
if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.setWindowTitle("Mesh Visualization")
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    sys.exit(app.exec_())
