import pyvista as pv
import sympy as sp
from sympy import Matrix, lambdify
import numpy as np
from PyQt5 import Qt, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from pyvistaqt import QtInteractor
import sys, os, time
import trimesh

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

        # create cubic skeleton
        self.cubic_skeleton_action = Qt.QAction('Cubic Skeleton', self)
        self.cubic_skeleton_action.triggered.connect(self.cubic_skeleton)
        editMenu.addAction(self.cubic_skeleton_action)
        
        if show:
            self.show()

        self.plotter.add_axes(interactive=None, line_width=2, color=None, x_color=None, y_color=None, z_color=None, xlabel='X', ylabel='Y', zlabel='Z', labels_off=False, box=None, box_args=None)

    def open_mesh(self):
        """ add a mesh to the pyqt frame """
        global mesh, mesh_vol

        # open file
        file_info = QtWidgets.QFileDialog.getOpenFileName()
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
        pre = trimesh.load(file_path)
        orient = pre.principal_inertia_transform
        pre = pre.apply_transform(orient)
        pre.export('data/'+ mesh_name + '_oriented.STL')
        mesh = pv.read('data/'+ mesh_name + '_oriented.STL')

        # print mesh info
        print("Mesh Name:", mesh_name)
        print("Mesh Type:", mesh_type[1:])

        # reset plotter
        self.reset_plotter()

        # find mesh centroid and translate the mesh so that's the origin
        # self.centroid()

        # mesh volume
        mesh_vol = float(format(mesh.volume, ".5f"))
        print("Mesh Volume:", mesh_vol)

    def reset_plotter(self):
        """ clear plotter of mesh or interactive options """
        # clear plotter
        self.plotter.clear()
        
        # callback opened mesh
        self.plotter.add_mesh(mesh, show_edges=True, color="w", opacity=0.6)
        
        # show origin
        self.plotter.add_axes_at_origin(xlabel='X', ylabel='Y', zlabel='Z', line_width=6, labels_off=True)
        
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
    
    def cubic_skeleton(self):
        ''' fill mesh with cubic skeleton'''
        # user input number of rays for next cubes
        self.plotter.add_text_slider_widget(self.max_cube_ray, ['1 rays','15 rays', '20 rays'], value=0)
        # self.plotter.add_text_slider_widget(self.next_cubes_ray, ['10 rays','15 rays', '20 rays'], value=0)
        regions = self.max_cube_slice()
        self.combine_corner_partitions(regions)
        
    def max_cube_ray(self, value):
        """ add a maximally inscribed cube within the opened mesh (via ray tracing) """
        global x_range, y_range, z_range, Vol_centroid
        global face_center, max_normal, max_cube_vol, max_cube
        global max_cube_start, max_cube_end, max_cube_run
        global top_rays, top_ints, bottom_rays, bottom_ints

        # # bypass error
        # try:
        #     max_cube, r_num
        # except NameError:
        #     max_cube = None
        #     r_num = 0

        # # remove old rays
        # if (r_num != 0) and (r_num == int(value[0])):
        #     return
        # elif (r_num != 0) and (max_cube_stored != None):
        #     self.plotter.remove_actor(max_cube_stored)

        # track starting time
        max_cube_start = time.time()

        # find mesh vertices
        V = np.array(mesh.points)

        # find the max and min of x,y,z axes of mesh
        ranges = mesh.bounds
        x_range = abs(ranges[0] - ranges[1])
        y_range = abs(ranges[2] - ranges[3])
        z_range = abs(ranges[4] - ranges[5])

        # show centroid
        Vol_centroid = np.array([0,0,0]) # overwrite centroid with origin at principle axes
        self.plotter.add_mesh(pv.PolyData(Vol_centroid), color='r', point_size=20.0, render_points_as_spheres=True)

        # find the nearest possible cube vertex from top rays & mesh intersection
        top_vert, top_rays, top_ints = self.cube_center_ray(Vol_centroid, 'z', value)
        top = self.furthest_pt(top_vert, Vol_centroid)

        # find the nearest possible cube vertex from bottom rays & mesh intersection
        bottom_vert, bottom_rays, bottom_ints = self.cube_center_ray(Vol_centroid, '-z', value)
        bottom = self.furthest_pt(bottom_vert, Vol_centroid)

        # find the nearest possible cube vertex between the two
        if top[0] < bottom[0]:
            p = top[1]
            V = top[2]
        else:
            p = bottom[1]
            V = bottom[2]
        
        # create and show max cube
        max_cube_V, max_cube_F, max_cube_vol = self.create_cube(V[p,:], Vol_centroid, np.array([0,0,Vol_centroid[2]]))
        max_cube = self.plotter.add_mesh(pv.PolyData(max_cube_V, max_cube_F), show_edges=True, line_width=3, color="g", opacity=0.6)
        max_cube_stored = max_cube

        # find & show max cube face centers
        cell_center = pv.PolyData(max_cube_V, max_cube_F).cell_centers()
        face_center = np.array(cell_center.points)
        #self.plotter.add_mesh(cell_center, color="r", point_size=8, render_points_as_spheres=True)

        # find max cube face normals
        max_normal = pv.PolyData(max_cube_V, max_cube_F).cell_normals

        # max cube volume
        max_cube_vol = float(format(max_cube_vol, ".5f"))
        print("Max Cube Volume:", max_cube_vol)

        # track ending time & duration
        max_cube_end = time.time()
        max_cube_run = max_cube_end - max_cube_start

        return

    def cube_center_ray(self, start, dir, value):
        ''' from starting point shoot out n rays to find vertices of possible cubes '''
        global r_num, r_rot, r_dec

        # initialize variables
        idx = value.index(" ")
        r_num = 0
        for i in range(0, idx):
            r_num = r_num + int(value[i]) + (idx - i)**10
        r_rot = np.pi/2
        r_dec = -2*np.pi/r_num
        l_wid = 5
        pt_size = 20
        ray_size = np.zeros((4, 3))
        r_dir = ray_size
        r_dir_norm = ray_size
        r_end = ray_size
        rays = [0] * r_num
        ints = [0] * r_num
        r_int = []
        ori_r_int = []

        # set ray length
        r_len = np.sqrt((x_range/2)**2 + (y_range/2)**2 + (z_range/2)**2)
        
        # create rays by rotating the first, which creates the cube with xyz axes as its face normals
        for i in range(0, r_num):
            for j in range(0, 4):
                if (j == 0) and (dir == 'z'):
                    r_dir[0] = np.array([np.sqrt(2)/2 * np.cos(np.pi/4 + r_dec * i), np.sqrt(2)/2 * np.sin(np.pi/4 + r_dec * i), 0.5])
                    r_dir_norm[0] = r_dir[0] / np.linalg.norm(r_dir[0])
                    r_end[0] = Vol_centroid + r_dir_norm[0] * r_len
                    # set rotation matrix about 'z'
                    R = self.rot_axis(np.array([0,0,1]))
                elif (j == 0) and (dir == '-z'):
                    r_dir[0] = np.array([np.sqrt(2)/2 * np.cos(np.pi/4 + r_dec * i), np.sqrt(2)/2 * np.sin(np.pi/4 + r_dec * i), -0.5])
                    r_dir_norm[0] = r_dir[0] / np.linalg.norm(r_dir[0])
                    r_end[0] = Vol_centroid + r_dir_norm[0] * r_len
                    # set rotation matrix about '-z'
                    R = self.rot_axis(np.array([0,0,-1]))
                else:
                    r_end[j] = np.dot(R(j*r_rot), (r_end[0]-Vol_centroid).T).T
                    r_end[j] = r_end[j] + Vol_centroid

                # perform ray trace
                r_pts, r_ind = mesh.ray_trace(Vol_centroid, r_end[j])

                # create an array of ray intersections
                r_int = np.append(r_int, r_pts[0])
            
            r_int = np.reshape(r_int, (4,3))
            ori_nearest, ori_p, ori_V = self.nearest_pt(r_int, Vol_centroid)
            r_int = []
            ori_r_int = np.append(ori_r_int, ori_V[ori_p,:])

        ori_r_int = np.reshape(ori_r_int, (r_num,3))
        return ori_r_int, rays, ints

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

    def furthest_pt(self, vert, starting_pt):
        global p, furthest, dist
        """ find furthest vertex among the list of nearest vertices """
        # find furthest point from the list of points
        c = len(vert)
        dist = np.zeros(c)
        for i in range(0, c):
            dist[i] = np.sqrt((vert[i,0] - starting_pt[0])**2 + (vert[i,1] - starting_pt[1])**2
                            + (vert[i,2] - starting_pt[2])**2)

        # find index of the furthest point
        furthest = max(dist)
        p = np.where(dist == furthest)
        p = p[0][0]

        return furthest, p, vert

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
        self.plotter.add_mesh(pv.PolyData(V_1), color="y", point_size=30.0, render_points_as_spheres=True)
        
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

        # cube volume
        cube_vol = (2 * np.linalg.norm(half_edge[0,:]))**3

        return cube_V, cube_F, cube_vol

    def rot_axis(self, axis):
        ''' create a rotational matrix about an arbitrary axis '''
        t = sp.Symbol('t')

        R_t = Matrix([[sp.cos(t)+axis[0]**2*(1-sp.cos(t)), axis[0]*axis[1]*(1-sp.cos(t))-axis[2]*sp.sin(t), axis[0]*axis[2]*(1-sp.cos(t))+axis[1]*sp.sin(t)],
            [axis[1]*axis[0]*(1-sp.cos(t))+axis[2]*sp.sin(t), sp.cos(t)+axis[1]**2*(1-sp.cos(t)), axis[1]*axis[2]*(1-sp.cos(t))-axis[0]*sp.sin(t)],
            [axis[2]*axis[0]*(1-sp.cos(t))-axis[1]*sp.sin(t), axis[2]*axis[1]*(1-sp.cos(t))+axis[0]*sp.sin(t), sp.cos(t)+axis[2]**2*(1-sp.cos(t))]])
        R = lambdify(t, R_t)
        return R

    def search_string_in_file(self, read_obj, file_name, string_to_search):
        """Search for the given string in file and return lines containing that string, along with line numbers"""
        loc = 0
        line_loc = []
        line_content = []
        a = []
        b = []
        c = []

        # close to change file access option
        read_obj.close()

        # Open the file in read only mode
        with open(file_name, 'r') as read_obj:
            # Read all lines in the file one by one
            for line in read_obj:
                # For each line, check if line contains the string
                loc += len(line)
                if string_to_search in line:
                    # If yes, then add the line location & line as a tuple in the list
                    line_loc.append(loc)
                    line_content.append(line.rstrip())
                    a = int(line_content[0][6])
                    b = int(line_content[0][8])
                    c = int(line_content[0][10])

        # Return list of tuples containing line locations and lines where string is found
        return a, b, c

    def max_cube_slice(self):
        ''' splitting the mesh in 27 regions according to the faces of max_cube
        and appending small, isolated partitions to the 27 main partitions '''
        global face_center
        # creating a 3x3x3 matrix representing the 27 regions
        height = np.zeros(3, dtype=object)
        side = np.zeros((3,3), dtype=object)
        cube = np.zeros((3,3,3), dtype=object)
        extra = np.zeros((3,3,3), dtype=object)

        # find face center x- and y-directions
        if abs(np.around(face_center[1][0],decimals=5)) != 0:
            if face_center[1][0] > face_center[3][0]:
                x_max = face_center[1]
                x_min = face_center[3]
            else:
                x_max = face_center[3]
                x_min = face_center[1]
            if face_center[2][1] > face_center[4][1]:
                y_max = face_center[2]
                y_min = face_center[4]
            else:
                y_max = face_center[4]
                y_min = face_center[2]
        elif abs(np.around(face_center[2][0],decimals=5)) != 0:
            if face_center[2][0] > face_center[4][0]:
                x_max = face_center[2]
                x_min = face_center[4]
            else:
                x_max = face_center[4]
                x_min = face_center[2]
            if face_center[1][1] > face_center[3][1]:
                y_max = face_center[1]
                y_min = face_center[3]
            else:
                y_max = face_center[3]
                y_min = face_center[1]
        
        # set face center z-directions
        z_max = face_center[0]
        z_min = face_center[5]

        # spliting the mesh along the z-axis
        height[0] = mesh.clip_closed_surface('z', origin=z_max).clean()
        height[1] = mesh.clip_closed_surface('-z', origin=z_max).clip_closed_surface('z', origin=z_min).clean()
        height[2] = mesh.clip_closed_surface('-z', origin=z_min).clean()

        # spliting the mesh along the y-axis
        for k in range(0,3):
            try:
                side[0,k] = height[k].clip_closed_surface('y', origin=y_max).clean()
                side[1,k] = height[k].clip_closed_surface('-y', origin=y_max).clip_closed_surface('y', origin=y_min).clean()
                side[2,k] = height[k].clip_closed_surface('-y', origin=y_min).clean()
            except ValueError:
                pass

        # splitting the mesh along the x-axis
        for j in range(0,3):
            for k in range(0,3):
                try:
                    cube[0,j,k] = side[j,k].clip_closed_surface('x', origin=x_max).clean()
                    cube[0,j,k] = cube[0,j,k].split_bodies()
                    
                    cube[1,j,k] = side[j,k].clip_closed_surface('x', origin=x_min).clip_closed_surface('-x', origin=x_max).clean() # need to test for knight
                    cube[1,j,k] = cube[1,j,k].split_bodies()
                    
                    cube[2,j,k] = side[j,k].clip_closed_surface('-x', origin=x_min).clean()
                    cube[2,j,k] = cube[2,j,k].split_bodies()
                except ValueError:
                    pass

        # clear plotter
        self.plotter.clear()

        # start output text file
        report = open("report.txt","w")
        print("Disconnected partitions:", end = "\n", file = report)

        # separate disconnected regions (extra[i,j,k])
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    if cube[i,j,k] != 0:
                        if cube[i,j,k].n_blocks > 1:
                            cube_num = cube[i,j,k].n_blocks
                            ratio = np.array([])
                            for w in range(1,cube_num):
                                ratio = np.append(ratio, np.divide(cube[i,j,k][0].volume, cube[i,j,k][w].volume))
                            if min(ratio) < 1:
                                p = np.where(ratio == min(ratio))
                                p = p[0][0] + 1
                            elif min(ratio) > 1:
                                p = 0
                            extra[i,j,k] = cube[i,j,k].copy()
                            cube[i,j,k] = pv.MultiBlock([cube[i,j,k]['Block-0' + str(p)]])
                            extra[i,j,k].pop(extra[i,j,k].keys()[p])
                            print("extra[", i, j, k, "][", extra[i,j,k].n_blocks - 1, "]", end = "\n", file = report)
                        else:
                            cube[i,j,k] = pv.MultiBlock([cube[i,j,k]['Block-00']])
                            extra[i,j,k] = pv.MultiBlock()
                    else:
                        extra[i,j,k] = pv.MultiBlock()

        print("\n", file = report)
        print("Proceed to append disconnected partitions to main partitions...", end = "\n", file = report)

        # merge the small disconnected regions (extra[i,j,k])
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    # extra[i,j,k].clean()
                    if extra[i,j,k].n_blocks != 0:
                        extra_num_1 = extra[i,j,k].n_blocks
                        l = 0
                        break_outter = False
                        while l < extra_num_1:
                            if (i == 0) or (i == 2):
                                # print(i,j,k, "\n", file = report)
                                if extra[i,j,k][l].merge(cube[1,j,k][0]).split_bodies().n_blocks == 1:
                                    cube[1,j,k].append(extra[i,j,k][l])
                                    print("cube[", 1, j, k, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[1,j,k] = pv.MultiBlock([cube[1,j,k].combine()])
                                    extra[i,j,k][l] = cube[1,j,k][0].copy()
                                    l += 1
                                    break
                                elif extra[1,j,k].n_blocks != 0:
                                    extra_num_2 = extra[1,j,k].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[1,j,k][m].volume
                                        if (extra[i,j,k][l].merge(extra[1,j,k][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[1,j,k][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[1,j,k][m])
                                                print("extra[", 1, j, k, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[1,j,k][m] = extra[1,j,k][m].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(1) + " " + str(j) + " " + str(k) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[1,j,k][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", 1, j, k, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
                            elif i == 1:
                                # print(i,j,k, "\n", file = report)
                                if extra[i,j,k][l].merge(cube[0,j,k][0]).split_bodies().n_blocks == 1:
                                    cube[0,j,k].append(extra[i,j,k][l])
                                    print("cube[", 0, j, k, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[0,j,k] = pv.MultiBlock([cube[0,j,k].combine()])
                                    extra[i,j,k][l] = cube[0,j,k][0].copy()
                                    l += 1
                                    break
                                elif extra[0,j,k].n_blocks != 0:
                                    extra_num_2 = extra[0,j,k].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[0,j,k][m].volume
                                        if (extra[i,j,k][l].merge(extra[0,j,k][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[0,j,k][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[0,j,k][m])
                                                print("extra[", 0, j, k, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[0,j,k][m] = extra[0,j,k][m].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(0) + " " + str(j) + " " + str(k) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[0,j,k][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", 0, j, k, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
                                elif extra[i,j,k][l].merge(cube[2,j,k][0]).split_bodies().n_blocks == 1:
                                    cube[2,j,k].append(extra[i,j,k][l])
                                    print("cube[", 2, j, k, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[2,j,k] = pv.MultiBlock([cube[2,j,k].combine()])
                                    extra[i,j,k][l] = cube[2,j,k][0].copy()
                                    l += 1
                                    break
                                elif extra[2,j,k].n_blocks != 0:
                                    extra_num_2 = extra[2,j,k].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[2,j,k][m].volume
                                        if (extra[i,j,k][l].merge(extra[2,j,k][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[2,j,k][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[2,j,k][m])
                                                print("extra[", 2, j, k, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[2,j,k][m] = extra[2,j,k][m].append(extra[i,j,k][l])
                                                partition = " extra[ " + str(2) + " " + str(j) + " " + str(k) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[2,j,k][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", 2, j, k, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break

                            if (j == 0) or (j == 2):
                                # print(i,j,k, "\n", file = report) 
                                if extra[i,j,k][l].merge(cube[i,1,k][0]).split_bodies().n_blocks == 1:
                                    cube[i,1,k].append(extra[i,j,k][l])
                                    print("cube[", i, 1, k, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[i,1,k] = pv.MultiBlock([cube[i,1,k].combine()])
                                    extra[i,j,k][l] = cube[i,1,k][0].copy()
                                    l += 1
                                    break
                                elif extra[i,1,k].n_blocks != 0:
                                    extra_num_2 = extra[i,1,k].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[i,1,k][m].volume
                                        if (extra[i,j,k][l].merge(extra[i,1,k][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[i,1,k][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[i,1,k][m])
                                                print("extra[", i, 1, k, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[i,1,k][m] = extra[i,1,k][m].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(i) + " " + str(1) + " " + str(k) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[i,1,k][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", i, 1, k, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
                            elif j == 1:
                                # print(i,j,k, "\n", file = report)
                                if extra[i,j,k][l].merge(cube[i,0,k][0]).split_bodies().n_blocks == 1:
                                    cube[i,0,k].append(extra[i,j,k][l])
                                    print("cube[", i, 0, k, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[i,0,k] = pv.MultiBlock([cube[i,0,k].combine()])
                                    extra[i,j,k][l] = cube[i,0,k][0].copy()
                                    l += 1
                                    break
                                elif extra[i,0,k].n_blocks != 0:
                                    extra_num_2 = extra[i,0,k].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[i,0,k][m].volume
                                        if (extra[i,j,k][l].merge(extra[i,0,k][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[i,0,k][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[i,0,k][m])
                                                print("extra[", i, 0, k, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[i,0,k][m] = extra[i,0,k][m].append(extra[i,j,k][l])
                                                partition = " extra[ " + str(i) + " " + str(0) + " " + str(k) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[i,0,k][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", i, 0, k, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
                                elif extra[i,j,k][l].merge(cube[i,2,k][0]).split_bodies().n_blocks == 1:
                                    cube[i,2,k].append(extra[i,j,k][l])
                                    print("cube[", i, 2, k, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[i,2,k] = pv.MultiBlock([cube[i,2,k].combine()])
                                    extra[i,j,k][l] = cube[i,2,k][0].copy()
                                    l += 1
                                    break
                                elif extra[i,2,k].n_blocks != 0:
                                    extra_num_2 = extra[i,2,k].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[i,2,k][m].volume
                                        if (extra[i,j,k][l].merge(extra[i,2,k][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[i,2,k][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[i,2,k][m])
                                                print("extra[", i, 2, k, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[i,2,k][m] = extra[i,2,k][m].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(i) + " " + str(2) + " " + str(k) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[i,2,k][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", i, 2, k, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break

                            if (k == 0) or (k == 2):
                                # print(i,j,k, "\n", file = report)
                                if extra[i,j,k][l].merge(cube[i,j,1][0]).split_bodies().n_blocks == 1:
                                    cube[i,j,1].append(extra[i,j,k][l])
                                    print("cube[", i, j, 1, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[i,j,1] = pv.MultiBlock([cube[i,j,1].combine()])
                                    extra[i,j,k][l] = cube[i,j,1][0].copy()
                                    # print(extra[i,j,k][l])
                                    l += 1
                                    break
                                elif extra[i,j,1].n_blocks != 0:
                                    extra_num_2 = extra[i,j,1].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[i,j,1][m].volume
                                        if (extra[i,j,k][l].merge(extra[i,j,1][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[i,j,1][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[i,j,1][m])
                                                print("extra[", i, j, 1, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[i,j,1][l] = extra[i,j,1][l].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(i) + " " + str(j) + " " + str(1) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[i,j,1][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", i, j, 1, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
                            elif k == 1:
                                # print(i,j,k, "\n", file = report)
                                if extra[i,j,k][l].merge(cube[i,j,0][0]).split_bodies().n_blocks == 1:
                                    cube[i,j,0].append(extra[i,j,k][l])
                                    print("cube[", i, j, 0, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[i,j,0] = pv.MultiBlock([cube[i,j,0].combine()])
                                    extra[i,j,k][l] = cube[i,j,0][0].copy()
                                    l += 1
                                    break
                                elif extra[i,j,0].n_blocks != 0:
                                    extra_num_2 = extra[i,j,0].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[i,j,0][m].volume
                                        if (extra[i,j,k][l].merge(extra[i,j,0][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[i,j,0][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[i,j,0][m])
                                                print("extra[", i, j, 0, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[i,j,0][m] = extra[i,j,0][m].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(i) + " " + str(j) + " " + str(0) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[i,j,0][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", i, j, 0, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
                                elif extra[i,j,k][l].merge(cube[i,j,2][0]).split_bodies().n_blocks == 1:
                                    cube[i,j,2].append(extra[i,j,k][l])
                                    print("cube[", i, j, 2, "] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                    cube[i,j,2] = pv.MultiBlock([cube[i,j,2].combine()])
                                    extra[i,j,k][l] = cube[i,j,2][0].copy()
                                    l += 1
                                    break
                                elif extra[i,j,2].n_blocks != 0:
                                    extra_num_2 = extra[i,j,2].n_blocks
                                    m = 0
                                    while m < extra_num_2:
                                        extra_sum = extra[i,j,k][l].volume + extra[i,j,2][m].volume
                                        if (extra[i,j,k][l].merge(extra[i,j,2][m]).split_bodies().n_blocks == 1) and (extra_sum >= mesh.volume / 1000):
                                            if extra[i,j,k][l].volume > extra[i,j,2][m].volume:
                                                extra[i,j,k][l] = extra[i,j,k][l].merge(extra[i,j,2][m])
                                                print("extra[", i, j, 2, "][", m ,"] --> extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                            else:
                                                extra[i,j,2][m] = extra[i,j,2][m].merge(extra[i,j,k][l])
                                                partition = " extra[ " + str(i) + " " + str(j) + " " + str(2) + " ][ " + str(m) + " ]"
                                                a, b, c = self.search_string_in_file(report, "report.txt", partition)
                                                report = open("report.txt", "a")
                                                if a != []:
                                                    cube[a,b,c].append(extra[i,j,2][m])
                                                    cube[a,b,c] = pv.MultiBlock([cube[a,b,c].combine()])
                                                print("extra[", i, j, 2, "][", m ,"] <-- extra[", i, j, k, "][", l ,"]", end = "\n", file = report)
                                                l += 1
                                                break_outter = True
                                                break
                                        else:
                                            m += 1
                                    if break_outter == True:
                                        break
        
        # close report
        report.close()

        # # section color choices
        # color = ["r", "b", "g", "y", "cyan"]
        # ind = -1

        # # display sections
        # for i in range(0,3):
        #     for j in range(0,3):
        #         for k in range(0,3):
        #             # rotate the 5 indicating colors
        #             if ind == 4:
        #                 ind = 0
        #             else:
        #                 ind += 1
        #             if cube[i,j,k]!= 0:
        #             # if extra[i,j,k].n_blocks!= 0:
        #                 self.plotter.add_mesh(cube[i,j,k], show_edges=True, color=color[ind], opacity=0.4)
        #                 # self.plotter.add_mesh(extra[i,j,k], show_edges=True, color=color[ind], opacity=0.4)
        #                 # filename = "section [" + str(i) + "," + str(j) + "," + str(k) +"].STL"
        #                 # cube[i,j,k].save("output/"+ filename)
    
        return cube

    def combine_corner_partitions(self, cube):
        ''' combine pairs of partitions at the corners to reduce total number of partitions '''
        # print to report
        report = open("report.txt", "a")
        print("\n", file = report)
        print("Proceed to combine corner partitions...", end = "\n", file = report)

        # indicate corner partition locations
        corner_ind = [0, 2]
        for o in corner_ind:
            for p in corner_ind:
                for q in corner_ind:
                    if (cube[1,p,q] != 0) and (cube[o,p,q][0].merge(cube[1,p,q][0]).split_bodies().n_blocks == 1):
                        o_pair = cube[o,p,q][0].merge(cube[1,p,q][0])
                        o_pair_vol = o_pair.volume
                        o_append = [1,p,q]
                    else:
                        o_pair = 0
                        o_pair_vol = 0
                        o_append = 0
                    if (cube[o,1,q] != 0) and (cube[o,p,q][0].merge(cube[o,1,q][0]).split_bodies().n_blocks == 1):
                        p_pair = cube[o,p,q][0].merge(cube[o,1,q][0])
                        p_pair_vol = p_pair.volume
                        p_append = [o,1,q]
                    else:
                        p_pair = 0
                        p_pair_vol = 0
                        p_append = 0
                    if (cube[o,p,1] != 0) and (cube[o,p,q][0].merge(cube[o,p,1][0]).split_bodies().n_blocks == 1):
                        q_pair = cube[o,p,q][0].merge(cube[o,p,1][0])
                        q_pair_vol = q_pair.volume
                        q_append = [o,p,1]
                    else:
                        q_pair = 0
                        q_pair_vol = 0
                        q_append = 0
                    pair_option = [o_pair, p_pair, q_pair]
                    append_option = [o_append, p_append, q_append]
                    pair_vol = np.array([o_pair_vol, p_pair_vol, q_pair_vol])
                    max_pair_vol = max(pair_vol)
                    if max_pair_vol != 0:
                        n = np.where(pair_vol == max_pair_vol)
                        n = n[0][0]
                        if n == 0:
                            print("cube[", o, p, q, "][ 0 ] <-- cube[", 1, p, q, "][ 0 ]", end = "\n", file = report)
                        elif n == 1:
                            print("cube[", o, p, q, "][ 0 ] <-- cube[", o, 1, q, "][ 0 ]", end = "\n", file = report)
                        elif n == 2:
                            print("cube[", o, p, q, "][ 0 ] <-- cube[", o, p, 1, "][ 0 ]", end = "\n", file = report)
                        cube[o,p,q] = pv.MultiBlock([pair_option[n]])
                        cube[append_option[n][0], append_option[n][1], append_option[n][2]] = 0
                    
        # section color choices
        color = ["r", "b", "g", "y", "cyan"]
        ind = -1

        # display sections
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    # rotate the 5 indicating colors
                    if ind == 4:
                        ind = 0
                    else:
                        ind += 1
                    if cube[i,j,k] != 0:
                        self.plotter.add_mesh(cube[i,j,k], show_edges=True, color=color[ind], opacity=0.8)
                        file_name = "section [" + str(i) + "," + str(j) + "," + str(k) +"].STL"
                        pv.save_meshio("output/"+ file_name, cube[i,j,k][0])
        

    def next_cubes_ray(self, value):
        ''' create cubes within the mesh from the face centers of the first cube'''
        global next_cube_vol, max_normal
        global next_rays, next_ints, next_cubes

        # find max cube
        self.max_cube_ray(value)

        # # bypass error
        # try:
        #     next_rays, next_ints, next_cubes, r_num
        # except NameError:
        #     next_cubes = None
        #     r_num = 0

        # # remove old rays
        # if (r_num != 0) and (r_num == int(value[0])):
        #     return
        # elif (r_num != 0) and (next_cubes != None):
        #     for i in range(0,6):
        #         self.plotter.remove_actor(next_cubes[i])

        # track starting time
        next_cube_start = time.time()

        # initiate variables
        next_cube_vol_sum = 0
        next_cubes = [0] * 6
        next_rays = [0] * 6 * r_num
        next_ints = [0] * 6 * r_num
        
        # fix max_normal
        normal = face_center[0] - Vol_centroid
        if (np.sign(normal[2]) != np.sign(max_normal[0,2])):
            max_normal =  np.negative(max_normal)

        # loop through all 6 faces of max cube
        for i in range(0, 6):
            # create rotaional matrix about max cube normals
            R = self.rot_axis(max_normal[i])

            # initialize variables
            ray_size = np.zeros((4, 3))
            r_dir = ray_size
            r_dir_norm = ray_size
            r_end = ray_size

            # initialize ray trace parameters
            l_wid = 3
            pt_size = 10
            r_len = np.sqrt((x_range/2)**2 + (y_range/2)**2 + (z_range/2)**2)
            r_int = []
            ori_r_int = []
            
            for j in range(0, r_num):
                for k in range(0, 4):
                    if k == 0:
                        if (i == 0) or (i == 5):
                            r_dir[0] = np.array([np.sqrt(2)/2 * np.cos(np.pi/4 + r_dec * j), np.sqrt(2)/2 * np.sin(np.pi/4 + r_dec * j), max_normal[i][2]])
                        else:
                            x,y = sp.symbols('x,y')
                            f = sp.Eq(max_normal[i][0]*x + max_normal[i][1]*y, 0)
                            g = sp.Eq(x**2 + y**2, 0.5**2)
                            inc = sp.solve([f,g],(x,y))
                            r_dir[0] = np.array(max_normal[i] + [inc[0][0], inc[0][1], 0.5])
                        r_dir_norm[0] = r_dir[0] / np.linalg.norm(r_dir[0])
                        r_end[0] = face_center[i] + r_dir_norm[0] * r_len
                        r_end[0] = np.dot(R(j*r_dec), (r_end[0]-Vol_centroid).T).T
                    else:
                        r_end[k] = np.dot(R(k*r_rot), (r_end[0]-Vol_centroid).T).T
                        r_end[k] = r_end[k] + Vol_centroid

                    # perform ray trace
                    r_pts, r_ind = mesh.ray_trace(face_center[i], r_end[k])

                    # show rays
                    # next_rays[i*r_num+k] = self.plotter.add_mesh(pv.Line(face_center[i], r_end[k]), color='w', line_width=l_wid)
                    # next_ints[i*r_num+k] = self.plotter.add_mesh(pv.PolyData(r_pts[0]), color='w', point_size=pt_size)

                    # create an array of ray intersections
                    r_int = np.append(r_int, r_pts[0])

                # find nearest vertice among the ray intersections
                r_int = np.reshape(r_int, (4,3))
                ori_nearest, ori_p, ori_V = self.nearest_pt(r_int, face_center[i])
                r_int = []
                ori_r_int = np.append(ori_r_int, ori_V[ori_p,:])

            ori_r_int = np.reshape(ori_r_int, (r_num,3))
            face = self.furthest_pt(ori_r_int, face_center[i])

            # create cube from nearest vertice
            next_cube_V, next_cube_F, next_cube_vol = self.create_cube(face[2][face[1],:], face_center[i], max_normal[i])
            next_cubes[i] = self.plotter.add_mesh(pv.PolyData(next_cube_V, next_cube_F), show_edges=True, line_width=3, color="g", opacity=0.6)

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

        return
    
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