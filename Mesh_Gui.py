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
import sys, os, time
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
        # self.max_cube_cone_action = Qt.QAction('Max Cube - Cone', self)
        # self.max_cube_cone_action.triggered.connect(self.max_cube_cone)
        # editMenu.addAction(self.max_cube_cone_action)

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
        self.next_cubes_action.triggered.connect(self.next_cubes_ray)
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
        global mesh, mesh_vol

        # open file
        file_info = QtWidgets.QFileDialog.getOpenFileName()
        print(file_info)
        file_path = file_info[0]
        
        # determine file type and if conversion needed
        file_dir, file_name = os.path.split(file_path)
        mesh_name, mesh_type = os.path.splitext(file_name)

        # convert mesh file type
        #if ext != ".vtk" or ext != ".VTK":
        #    mesh = meshio.read(file_path)
        #    meshio.write(root + ".vtk", mesh)
        #    mesh = pv.read(head + "/" + root + ".vtk")
            # need to store elsewhere or delete .vtk file in the future
        #else:
        #    mesh = pv.read(file_path)

        # read mesh & transform according to principal axes
        pre = trimesh.load_mesh(file_path)
        orient = pre.principal_inertia_transform
        pre = pre.apply_transform(orient)
        pre.export('data/'+ mesh_name + '_oriented.STL')
        mesh = pv.read('data/'+ mesh_name + '_oriented.STL')

        # print mesh info
        print("Mesh Name:", mesh_name)
        print("Mesh Type:", mesh_type[1:])

        # show transformed mesh
        self.plotter.add_mesh(mesh, show_edges=True, color="w", opacity=0.6)

        # reset plotter
        self.reset_plotter()

        # find mesh centroid and translate the mesh so that's the origin
        self.centroid()

        # show bounding box
        # self.plotter.add_bounding_box(opacity=0.5, color="y")

        # mesh volume
        mesh_vol = float(format(mesh.volume, ".5f"))
        print("Mesh Volume:", mesh_vol)

    def reset_plotter(self):
        """ clear plotter of mesh or interactive options """
        # clear plotter
        self.plotter.clear()
        self.plotter.clear_plane_widgets()
        self.plotter.reset_camera()
        
        # callback opened mesh
        self.plotter.add_mesh(mesh, show_edges=True, color="w", opacity=0.6)
        
        # show origin
        self.plotter.add_axes_at_origin(xlabel='X', ylabel='Y', zlabel='Z', line_width=6, labels_off=True)

        # show floors
        #self.plotter.add_floor('-y')
        #self.plotter.add_floor('-z')
        
    def centroid(self):
        """ find centroid volumetrically and indicate on graph """
        global Vol_centroid, V

        # find the vertices & the vertex indices of each triangular face
        V = np.array(mesh.points)
        col = len(V)
        f_ind = np.array(mesh.faces.reshape((-1,4))[:, 1:4])
        
        # define an arbitrary start point from middle of max and min of X,Y,Z of
        # all points: in a convex manifold it falls inside the volume (requires
        # segmentation for general application)
        start = np.array(mesh.center)
        X_start = start[0]
        Y_start = start[1]
        Z_start = start[2]
        
        # initialize variables
        centroids = []
        Vol_total = 0
        Sum_vol_x = 0
        Sum_vol_y = 0
        Sum_vol_z = 0
        
        # find centroid from all tetrahedra made with arbitrary center and triangular faces
        for i in range(0, col-1, 3):          
            # find the center of each tetrahedron (average of X,Y,Z of 
            # 4 vertices, 3 from the triangle, and one arbitrary start point)
            X_cent = (X_start + V[f_ind[i,0],0] + V[f_ind[i+1,0],0] + V[f_ind[i+2,0],0]) / 4
            Y_cent = (Y_start + V[f_ind[i,1],1] + V[f_ind[i+1,1],1] + V[f_ind[i+2,1],1]) / 4
            Z_cent = (Z_start + V[f_ind[i,2],2] + V[f_ind[i+1,2],2] + V[f_ind[i+2,2],2]) / 4
    
            # compute the volume of each tetrahedron
            V1 = np.array([V[f_ind[i,0],0], V[f_ind[i,1],1], V[f_ind[i,2],2]])**2 - np.array([X_start, Y_start, Z_start])**2
            V2 = np.array([V[f_ind[i+1,0],0], V[f_ind[i+1,1],1], V[f_ind[i+1,2],2]])**2 - np.array([V[f_ind[i,0],0], V[f_ind[i,1],1], V[f_ind[i,2],2]])**2
            V3 = np.array([V[f_ind[i+2,0],0], V[f_ind[i+2,1],1], V[f_ind[i+2,2],2]])**2 - np.array([V[f_ind[i+1,0],0], V[f_ind[i+1,1],1], V[f_ind[i+1,2],2]])**2
            V1 = V1.reshape((-1,1))
            V2 = V2.reshape((-1,1))
            V3 = V3.reshape((-1,1))
            Vol = abs(np.linalg.det(np.hstack([V1, V2, V3]))) / 6
    
            # tally up each cycle
            Vol_total = Vol_total + Vol
            Sum_vol_x = Sum_vol_x + Vol * X_cent
            Sum_vol_y = Sum_vol_y + Vol * Y_cent
            Sum_vol_z = Sum_vol_z + Vol * Z_cent
            centroids.append([X_cent,Y_cent,Z_cent])
        
        # find & show centroid
        centroids = np.asarray(centroids)
        Vol_centroid = [Sum_vol_x, Sum_vol_y, Sum_vol_z] / Vol_total
        # print("Total Volume:", Vol_total)
        # print("Centroid:", Vol_centroid)

    # def max_cube_cone(self):
    #     """ add a maximally inscribed cube within the opened mesh (via cone intersection) """
    #     global ranges, max_c1, max_c2, nearest_vert
    #     global face_center, max_cube, max_normal, max_cube_vol
    #     global cube_V, cube_F
    #     global Vol_centroid, V

    #     # find the vertices & the vertex indices of each triangular face
    #     V = np.array(mesh.points)

    #     # reset plotter
    #     self.reset_plotter()

    #     # show centroid
    #     #Vol_centroid = np.array([0,0,0])
    #     self.plotter.add_mesh(pv.PolyData(Vol_centroid), color='r', point_size=20.0, render_points_as_spheres=True)

    #     # project cones to from centroid to find maximally inscribed cube
    #     ranges = mesh.bounds
    #     h = (abs(ranges[4]) + abs(ranges[5]))/3
    #     ang = np.arctan(0.5/(np.sqrt(2)/2))
    #     ang = float(90 - np.degrees(ang))
    #     max_c1 = pv.Cone(center=Vol_centroid+[0,0,h/2], direction=[0,0,-1], height=h, radius=None, capping=False, angle=ang, resolution=100)
    #     max_c2 = pv.Cone(center=Vol_centroid-[0,0,h/2], direction=[0,0,1], height=h, radius=None, capping=False, angle=ang, resolution=100)
    #     self.plotter.add_mesh(max_c1, color="r", show_edges=True, opacity=0.4)
    #     self.plotter.add_mesh(max_c2, color="r", show_edges=True, opacity=0.4)
        
    #     # find the nearest possible cube vertex from top cone & mesh intersection
    #     top_clip = mesh.clip_surface(max_c1, invert=True)
    #     top_vert = np.array(top_clip.points)
    #     top = self.nearest_pt(top_vert, Vol_centroid)

    #     # find the nearest possible cube vertex from bottom cone & mesh intersection
    #     bottom_clip = mesh.clip_surface(max_c2, invert=True)
    #     bottom_vert = np.array(bottom_clip.points)
    #     bottom = self.nearest_pt(bottom_vert, Vol_centroid)

    #     # show top & bottom clipped surfaces
    #     #self.plotter.add_mesh(top_clip, opacity=0.6, show_edges=True, color="g")
    #     #self.plotter.add_mesh(bottom_clip, opacity=0.6, show_edges=True, color="g")

    #     # find the nearest possible cube vertex between the two
    #     if top[0] < bottom[0]:
    #         p = top[1]
    #         V = top[2]
    #     else:
    #         p = bottom[1]
    #         V = bottom[2]
        
    #     # create max cube from nearest possible cube vertex
    #     cube_V, cube_F = self.create_cube(V[p,:], Vol_centroid, np.array([0,0,1]))
    #     max_cube = pv.PolyData(cube_V, cube_F)

    #     # show max cube
    #     self.plotter.add_mesh(max_cube, show_edges=True, color="b", opacity=0.6)

    #     # record nearest vertex
    #     nearest_vert = V[p,:]

    #     # find & show max cube face centers
    #     cell_center = max_cube.cell_centers()
    #     face_center = np.array(cell_center.points)
    #     self.plotter.add_mesh(cell_center, color="r", point_size=8.0, render_points_as_spheres=True)

    #     # find max cube face normals
    #     max_normal = max_cube.cell_normals

    #     # max cube volume
    #     max_cube_vol = float(format(max_cube.volume, ".5f"))
    #     print("Max Cube Volume:", max_cube_vol)

    def max_cube_ray(self):
        """ add a maximally inscribed cube within the opened mesh (via ray tracing) """
        global ranges, nearest_vert
        global face_center, max_cube, max_normal, max_cube_vol
        global max_cube_V, max_cube_F
        global Vol_centroid, V
        global max_cube_start, max_cube_end, max_cube_run

        # track starting time
        max_cube_start = time.time()

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
        top_vert = self.cube_center_ray(Vol_centroid, 'z')
        top = self.nearest_pt(top_vert, Vol_centroid)

        # find the nearest possible cube vertex from bottom rays & mesh intersection
        bottom_vert = self.cube_center_ray(Vol_centroid, '-z')
        bottom = self.nearest_pt(bottom_vert, Vol_centroid)

        # find the nearest possible cube vertex between the two
        if top[0] < bottom[0]:
            p = top[1]
            V = top[2]
        else:
            p = bottom[1]
            V = bottom[2]
        
        # create and show max cube
        max_cube_V, max_cube_F, max_cube_vol = self.create_cube(V[p,:], Vol_centroid, np.array([0,0,Vol_centroid[2]]))
        max_cube = pv.PolyData(max_cube_V, max_cube_F)
        self.plotter.add_mesh(max_cube, show_edges=True, line_width=3, color="g", opacity=0.6)

        # record nearest vertex
        nearest_vert = V[p,:]

        # find & show max cube face centers
        cell_center = max_cube.cell_centers()
        face_center = np.array(cell_center.points)
        print("Max Cube Face Centers:", face_center)
        self.plotter.add_mesh(cell_center, color="r", point_size=8, render_points_as_spheres=True)

        # find max cube face normals
        max_normal = max_cube.cell_normals
        print("Max Cube Normals:", max_normal)

        # max cube volume
        max_cube_vol = float(format(max_cube_vol, ".5f"))
        print("Max Cube Volume:", max_cube_vol)

        # track ending time & duration
        max_cube_end = time.time()
        max_cube_run = max_cube_end - max_cube_start

    def cube_center_ray(self, start, dir):
        ''' from starting point shoot out 8 rays to find vertices of a possible cube,
        whose face normals would be in either in x,y,z direction, or rotated 45 deg along z-axis '''
        # set ray directions
        if dir == 'z':
            r1_dir = np.array([1,1,1])
            #r2_dir = np.array([1,0,1])
            r3_dir = np.array([1,-1,1])
            #r4_dir = np.array([0,-1,1])
            r5_dir = np.array([-1,-1,1])
            #r6_dir = np.array([-1,0,1])
            r7_dir = np.array([-1,1,1])
            #r8_dir = np.array([0,1,1])
        elif dir == '-z':
            r1_dir = np.array([1,1,-1])
            #r2_dir = np.array([1,0,-1])
            r3_dir = np.array([1,-1,-1])
            #r4_dir = np.array([0,-1,-1])
            r5_dir = np.array([-1,-1,-1])
            #r6_dir = np.array([-1,0,-1])
            r7_dir = np.array([-1,1,-1])
            #r8_dir = np.array([0,1,-1])

        # set ray length
        r_len = abs(ranges[4] - ranges[5])/2 * np.sqrt(0.5**2 + (np.sqrt(2)/2)**2)

        # set ray end points
        r1_end = start + r1_dir / np.linalg.norm(r1_dir) * r_len
        #r2_end = start + r2_dir / np.linalg.norm(r2_dir) * r_len
        r3_end = start + r3_dir / np.linalg.norm(r3_dir) * r_len
        #r4_end = start + r4_dir / np.linalg.norm(r4_dir) * r_len
        r5_end = start + r5_dir / np.linalg.norm(r5_dir) * r_len
        #r6_end = start + r6_dir / np.linalg.norm(r6_dir) * r_len
        r7_end = start + r7_dir / np.linalg.norm(r7_dir) * r_len
        #r8_end = start + r8_dir / np.linalg.norm(r8_dir) * r_len
        
        # perform ray trace
        r1_pts, r1_ind = mesh.ray_trace(start, r1_end)
        #r2_pts, r2_ind = mesh.ray_trace(start, r2_end)
        r3_pts, r3_ind = mesh.ray_trace(start, r3_end)
        #r4_pts, r4_ind = mesh.ray_trace(start, r4_end)
        r5_pts, r5_ind = mesh.ray_trace(start, r5_end)
        #r6_pts, r6_nd = mesh.ray_trace(start, r6_end)
        r7_pts, r7_ind = mesh.ray_trace(start, r7_end)
        #r8_pts, r8_ind = mesh.ray_trace(start, r8_end)

        # initialize rays
        r1 = pv.Line(start, r1_end)
        #r2 = pv.Line(start, r2_end)
        r3 = pv.Line(start, r3_end)
        #r4 = pv.Line(start, r4_end)
        r5 = pv.Line(start, r5_end)
        #r6 = pv.Line(start, r6_end)
        r7 = pv.Line(start, r7_end)
        #r8 = pv.Line(start, r8_end)

        # initialize intersections
        r1_int = pv.PolyData(r1_pts[0])
        #r2_int = pv.PolyData(r2_pts[0])
        r3_int = pv.PolyData(r3_pts[0])
        #r4_int = pv.PolyData(r4_pts[0])
        r5_int = pv.PolyData(r5_pts[0])
        #r6_int = pv.PolyData(r6_pts[0])
        r7_int = pv.PolyData(r7_pts[0])
        #r8_int = pv.PolyData(r8_pts[0])

        # show rays
        l_wid = 6
        #self.plotter.add_mesh(r1, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r2, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r3, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r4, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r5, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r6, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r7, color='w', line_width=l_wid)
        #self.plotter.add_mesh(r8, color='w', line_width=l_wid)

        # show intersections
        pt_size = 20
        #self.plotter.add_mesh(r1_int, color='w', point_size=pt_size)
        #self.plotter.add_mesh(r2_int, color='w', point_size=pt_size)
        #self.plotter.add_mesh(r3_int, color='w', point_size=pt_size)
        #self.plotter.add_mesh(r4_int, color='w', point_size=pt_size)
        #self.plotter.add_mesh(r5_int, color='w', point_size=pt_size)
        #self.plotter.add_mesh(r6_int, color='w', point_size=pt_size)
        #elf.plotter.add_mesh(r7_int, color='w', point_size=pt_size)
        #self.plotter.add_mesh(r8_int, color='w', point_size=pt_size)

        # array of intersections
        # r_int = np.vstack([r1_int.points, r2_int.points, r3_int.points, r4_int.points,
        #             r5_int.points, r6_int.points, r7_int.points, r8_int.points])
        r_int = np.vstack([r1_int.points, r3_int.points, r5_int.points, r7_int.points])

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
            
    def rot_axis(self, axis):
        ''' create a rotational matrix about an arbitrary axis '''
        t = sp.Symbol('t')

        R_t = Matrix([[sp.cos(t)+axis[0]**2*(1-sp.cos(t)), axis[0]*axis[1]*(1-sp.cos(t))-axis[2]*sp.sin(t), axis[0]*axis[2]*(1-sp.cos(t))+axis[1]*sp.sin(t)],
            [axis[1]*axis[0]*(1-sp.cos(t))+axis[2]*sp.sin(t), sp.cos(t)+axis[1]**2*(1-sp.cos(t)), axis[1]*axis[2]*(1-sp.cos(t))-axis[0]*sp.sin(t)],
            [axis[2]*axis[0]*(1-sp.cos(t))-axis[1]*sp.sin(t), axis[2]*axis[1]*(1-sp.cos(t))+axis[0]*sp.sin(t), sp.cos(t)+axis[2]**2*(1-sp.cos(t))]])
        R = lambdify(t, R_t)
        return R

    def create_cube(self, vertex, starting_pt, axis):
        ''' create cube from the nearest pt & centroid '''
        if (axis[0] == 0) and (axis[1] == 0) and (axis[2] == 0):
            axis[2] = 1
            vert_trans = np.array([0,0,0])
        elif (Vol_centroid[0] == 0) and (Vol_centroid[1] == 0) and (Vol_centroid[2] == 0):
            vert_trans = np.array([0,0,0])
        else:
            vert_trans = starting_pt
            for i in range(0,3):
                if round(axis[i]) == 1 or round(axis[i]) == -1:
                    vert_trans[i] == 0
        # find the other 7 vertices
        # 3 vertices can be found by rotating the first point 90 degrees 3 times around Z axis of centroid
        # 4 vertices can be found by translating the first four vertices twice the half edge
        # found from the distance times sin(pi/4)
        R = self.rot_axis(axis / np.linalg.norm(axis))
        
        # construct the array of the first 4 vertices
        V_1 = np.array(vertex - vert_trans)
        V_2 = np.dot(R(np.pi/2), V_1.T).T
        V_3 = np.dot(R(np.pi), V_1.T).T
        V_4 = np.dot(R(3*np.pi/2), V_1.T).T
        # cube_V_start = np.array([V_1, V_2, V_3, V_4])
        cube_V_start = np.array([V_1, V_2, V_3, V_4]) + np.ones((4,1)) * [vert_trans]
        cube_V_start_center = np.array(pv.PolyData(cube_V_start).center)

        # show nearest vertex of cube
        V_1 = np.array(vertex)
        #self.plotter.add_mesh(pv.PolyData(V_1), color="y", point_size=30.0, render_points_as_spheres=True)
        
        # find the translation distance
        trans_dis = starting_pt - cube_V_start_center
        trans_dir = trans_dis / np.linalg.norm(trans_dis)
        dia_dis = np.sqrt((V_1[0]-cube_V_start_center[0])**2 + (V_1[1]-cube_V_start_center[1])**2 + (V_1[2]-cube_V_start_center[2])**2)
        half_edge = np.ones((4,1)) * [trans_dir] * dia_dis * np.sin(np.pi/4)
        cube_trans = np.asarray(2*half_edge, dtype=np.float64)

        # construct the cube
        cube_V_end = np.add(cube_V_start, cube_trans)
        cube_V = np.vstack((cube_V_start, cube_V_end))
        cube_F = np.hstack([[4,0,1,2,3],
                        [4,0,3,7,4],
                        [4,0,1,5,4],
                        [4,1,2,6,5],
                        [4,2,3,7,6],
                        [4,4,5,6,7]])

        # test for cube-ness
        # face_edge = np.sqrt((V_1[0]-V_2[0])**2 + (V_1[1]-V_2[1])**2 + (V_1[2]-V_2[2])**2)
        # trans_edge = np.sqrt((V_1[0]-cube_V_end[0][0])**2 + (V_1[1]-cube_V_end[0][1])**2 + (V_1[2]-cube_V_end[0][2])**2)
        # if round(face_edge, 4) == round(trans_edge, 4):
        #     print('Face Edge = ', round(face_edge, 4))
        #     print('Translation Edge = ', round(trans_edge, 4))
        #     print('Cube\n')
        # else:
        #     print('Face Edge = ', round(face_edge, 4))
        #     print('Translation Edge = ', round(trans_edge, 4))
        #     print('Not Cube\n')

        # cube volume
        cube_vol = (2 * np.linalg.norm(half_edge[0,:]))**3

        return cube_V, cube_F, cube_vol
    
    def ext_max_cube(self):
        ''' extend max cube into maximally inscribed cuboid '''
        global face_center, ext_max_cube, max_normal, ext_max_cube_vol

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
            np.reshape(max_cube_F, (6,5))
            faces = np.reshape(max_cube_F, (6,5))
            V_ind = faces[F_ind][0,1:5]
            current_V = np.vstack([max_cube_V[V_ind[0]], max_cube_V[V_ind[1]], max_cube_V[V_ind[2]], max_cube_V[V_ind[3]]])
            ext_V = self.ext_ray(current_V, ext_dir[i])
            max_cube_V[V_ind] = ext_V

        # create & show extended max cube
        ext_max_cube = pv.PolyData(max_cube_V, max_cube_F)
        self.plotter.add_mesh(ext_max_cube, show_edges=True, color="y", opacity=0.6)

        # find face centers of extended max cube
        cell_center = ext_max_cube.cell_centers()
        face_center = np.array(cell_center.points)

        # find face normals of the extended max cube
        max_normal = ext_max_cube.cell_normals

        # extended max cube volume
        ext_max_cube_vol = float(format(ext_max_cube.volume, ".5f"))
        print("Extended Max Cube Volume:", ext_max_cube_vol)

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

    def next_cubes_ray(self):
        ''' create cubes within the mesh from the face centers of the first cube'''
        global next_cube_vol, max_normal

        # track starting time
        next_cube_start = time.time()
        
        # initiate variable
        next_cube_vol_sum = 0

        # fix max_normal
        normal = face_center[0] - Vol_centroid
        if (np.sign(normal[2]) != np.sign(max_normal[0,2])):
            max_normal =  np.negative(max_normal)

        # loop through all 6 faces of max cube
        for i in range(0, 6):
            # create rotaional matrix about max cube normals
            R = self.rot_axis(max_normal[i])
        
            # initialize variables (8 rays)
            # r_dir = np.zeros((8, 3))
            # r_dir_norm = np.zeros((8, 3))
            # r_end = np.zeros((8, 3))

            # initialize variables (4 rays)
            ray_size = np.zeros((4, 3))
            r_dir = ray_size
            #r_flat_dir = ray_size
            #r_flat_dir_norm = ray_size
            r_dir_norm = ray_size
            r_end = ray_size
            #r_flat_end = ray_size

            # initialize ray trace parameters
            l_wid = 5
            pt_size = 20
            x_range = abs(ranges[0] - ranges[1])
            y_range = abs(ranges[2] - ranges[3])
            z_range = abs(ranges[4] - ranges[5])
            r_len = np.sqrt((x_range/2)**2 + (y_range/2)**2 + (z_range/2)**2) * np.sqrt(1**2 + (np.sqrt(2)/2)**2)
            r_int = np.array([])

            # create a set of 8 rays from each face
            # for j in range(0, 8):
            #     if j == 0:
            #         if (i == 0) or (i == 5):
            #             r_dir[j] = np.array(max_normal[i] + [0.5,0.5,0])
            #             r_dir_norm[j] = r_dir[j] / np.linalg.norm(r_dir[j])
            #             r_end[j] = face_center[i] + r_dir_norm[j] * r_len_z
            #         else:
            #             r_dir[j] = np.array(max_normal[i] + [0,0,np.sqrt(2)/2])
            #             r_dir_norm[j] = r_dir[j] / np.linalg.norm(r_dir[j])
            #             r_end[j] = face_center[i] + r_dir_norm[j] * r_len_xy
            #     else:
            #         r_dir[j] = np.dot(R(j*np.pi/4), r_dir[0].T).T
            #         r_dir_norm[j] = r_dir[j] / np.linalg.norm(r_dir[j])
            #         if (i == 0) or (i == 5):
            #             r_end[j] = face_center[i] + r_dir_norm[j] * r_len_z
            #         else:
            #             r_end[j] = face_center[i] + r_dir_norm[j] * r_len_xy
                
            #     # perform ray trace
            #     r_pts, r_ind = mesh.ray_trace(face_center[i], r_end[j])

            #     # show rays
            #     self.plotter.add_mesh(pv.Line(face_center[i], r_end[j]), color='y', line_width=l_wid)
            #     self.plotter.add_mesh(pv.PolyData(r_pts[0]), color='y', point_size=pt_size)

            #     # create an array of ray intersections
            #     r_int = np.append(r_int, r_pts[0])
            
            for j in range(0, 4):
                if j == 0:
                    r_dir[0] = np.array(max_normal[i] + ([1,1,1] - abs(max_normal[i])) / 2)
                    r_dir_norm[0] = r_dir[0] / np.linalg.norm(r_dir[0])
                    r_end[0] = face_center[i] + r_dir_norm[0] * r_len
                else:
                    r_end[j] = np.dot(R(j*np.pi/2), (r_end[0]-Vol_centroid).T).T
                    r_end[j] = r_end[j] + Vol_centroid

                # max_cube_faces = np.reshape(max_cube_F, (6,5))
                # r_flat_dir[j] = max_cube_V[max_cube_faces[i,j+1]] - face_center[i]

                # perform ray trace
                r_pts, r_ind = mesh.ray_trace(face_center[i], r_end[j])

                # r_flat_dir_norm[j] = r_flat_dir[j] / np.linalg.norm(r_flat_dir[j])
                # r_flat_end[j] = face_center[i] + r_flat_dir_norm[j] * r_len
                # r_flat_pts, r_flat_ind = mesh.ray_trace(face_center[i], r_flat_end[j])

                # show rays
                # self.plotter.add_mesh(pv.Line(face_center[i], r_end[j]), color='w', line_width=l_wid)
                # self.plotter.add_mesh(pv.PolyData(r_pts[0]), color='w', point_size=pt_size)
                
                # self.plotter.add_mesh(pv.Line(face_center[i], r_flat_end[j]), color='w', line_width=l_wid)
                # self.plotter.add_mesh(pv.PolyData(r_flat_pts[0]), color='w', point_size=pt_size)

                # create an array of ray intersections
                r_int = np.append(r_int, r_pts[0])
                # r_int = np.append(r_int, r_flat_pts[0])

            # find nearest vertice among the ray intersections
            r_int = np.reshape(r_int, (4,3))
            r = self.nearest_pt(r_int, face_center[i])

            # create cube from nearest vertice
            next_cube_V, next_cube_F, next_cube_vol = self.create_cube(r[2][r[1],:], face_center[i], max_normal[i])
            next_cube = pv.PolyData(next_cube_V, next_cube_F)
            self.plotter.add_mesh(next_cube, show_edges=True, line_width=3, color="g", opacity=0.6)
            
            # show cut
            # u1 = next_cube.triangulate().boolean_union(mesh)
            # u2 = u1.boolean_difference(next_cube.triangulate())
            #diff = next_cube.delaunay_3d().clip_surface(mesh, invert=False)
            #self.plotter.clear()
            #self.plotter.add_mesh(next_cube.triangulate(), show_edges=True, line_width=3, color="g", opacity=0.6)
            #self.plotter.add_mesh(diff, show_edges=True, line_width=3, color="y", opacity=0.6)

            # testing
            # test_cube_V = next_cube_V[:-4,:]
            # test_cube_F = np.array([4,0,1,2,3])
            # test_cube = pv.PolyData(test_cube_V, test_cube_F)
            # self.plotter.add_mesh(test_cube, show_edges=True, line_width=3, color="g", opacity=0.6)
            
            # show next cube
            #self.plotter.add_mesh(next_cube, show_edges=True, line_width=3, color="g", opacity=0.6)

            # next cube volume
            next_cube_vol_sum = next_cube_vol_sum + next_cube_vol
        
        # show packing efficiency
        next_cube_vol_sum = float(format(next_cube_vol_sum, ".5f"))
        pack_vol = float(format((max_cube_vol + next_cube_vol_sum), ".5f"))
        pack_percent = "{:.1%}".format(pack_vol / mesh_vol)
        print("Next Cubes Volume:", next_cube_vol_sum)
        print("Packed Volume:", pack_vol)
        print("Packing Efficiency:", pack_percent)

        # track starting time
        next_cube_end = time.time()
        next_cube_run = next_cube_end - next_cube_start
        print("Total elapsed run time: %g seconds" % (max_cube_run + next_cube_run))


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

        # volume test
        chunck_vol = format(part2.volume, ".5f")
        print(chunck_vol)

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
