import pyvista as pv
import sympy as sp
from sympy import Matrix, lambdify
import numpy as np
from PyQt5 import Qt, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from pyvistaqt import QtInteractor
import sys, os, time
import trimesh
import pymeshfix as mf

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

        # set centroid
        self.skewed_centroid_action = Qt.QAction('Skewed Centroid', self)
        self.skewed_centroid_action.triggered.connect(self.skewed_centroid_check)
        fileMenu.addAction(self.skewed_centroid_action)
        
        # save screenshot
        self.save_screenshot_action = Qt.QAction('Save Screenshot', self)
        self.save_screenshot_action.triggered.connect(self.save_screenshot)
        fileMenu.addAction(self.save_screenshot_action)

        # exit button
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)
        
        # Show max cube & raytracing process
        self.max_cube_action = Qt.QAction('Max Cube', self)
        self.max_cube_action.triggered.connect(self.show_max_cube)
        editMenu.addAction(self.max_cube_action)

        # Show max cube & raytracing process
        self.max_cuboid_action = Qt.QAction('Max Cuboid', self)
        self.max_cuboid_action.triggered.connect(self.max_cuboid)
        editMenu.addAction(self.max_cuboid_action)
        
        if show:
            self.show()
        
        self.plotter.add_axes(interactive=None, line_width=2, x_color=None, y_color=None, z_color=None, xlabel='X', ylabel='Y', zlabel='Z', labels_off=False, box= None, box_args=None)
    
    def open_mesh(self):
        """ add a mesh to the pyqt frame """
        global int_surface, ext_surface, mesh_vol, mesh
        global x_range, y_range, z_range, Vol_centroid
        global open_mesh_run
        global mesh_name

        # track pre-processing starting time
        open_mesh_start = time.time()

        # open file
        file_info = QtWidgets.QFileDialog.getOpenFileName()
        file_path = file_info[0]
        
        # determine file type and if conversion needed
        _, file_name = os.path.split(file_path)
        mesh_name, mesh_type = os.path.splitext(file_name)

        # read mesh & transform according to principal axes
        print(file_path)
        pre = trimesh.load(file_path)
        orient = pre.principal_inertia_transform
        pre = pre.apply_transform(orient)
        post_file_path = 'data/'+ mesh_name + '_oriented.stl'
        pre.export(post_file_path)
        ext_surface = pv.read(post_file_path)

        # scale meshes accordingly
        if mesh_name == 'elephant':
            ext_surface.points *= 12  # Elephant
        elif mesh_name == 'Bracket S24D1':
            ext_surface.points /= 10  # Bracket
        elif mesh_name == 'knight':
            ext_surface.points /= 2 # Knight

        # create internal offset
        thickness = 0.1 # inches
        grid = pv.create_grid(ext_surface).triangulate()
        solid = grid.clip_surface(ext_surface)
        solid.compute_implicit_distance(ext_surface, inplace=True)
        imp_dis_max = max(solid['implicit_distance'])

        shell_threshold = imp_dis_max - thickness
        shell = solid.clip_scalar('implicit_distance', value = shell_threshold)
        int_surface = shell.extract_geometry()

        meshfix = mf.MeshFix(int_surface)
        meshfix.repair(verbose=True)

        mesh = solid.clip_surface(int_surface, invert=False)
        
        # print mesh info
        print("Mesh Name:", mesh_name)
        print("Mesh Type:", mesh_type[1:])

        # find mesh centroid and translate the mesh so that's the origin
        Vol_centroid = np.array([0,0,0])
        self.skewed_centroid_action.setCheckable(True)

        # reset plotter
        self.reset_plotter(Vol_centroid)

        # find the max and min of x,y,z axes of mesh
        ranges = mesh.bounds
        x_range = abs(ranges[0] - ranges[1])
        y_range = abs(ranges[2] - ranges[3])
        z_range = abs(ranges[4] - ranges[5])
        print("x:", float(format(x_range, ".2f")), "in")
        print("y:", float(format(y_range, ".2f")), "in")
        print("z:", float(format(z_range, ".2f")), "in")

        # mesh volume
        mesh_vol = float(format(mesh.volume, ".2f"))
        print("Mesh Volume:", mesh_vol, "in^3")

        # track pre-processing ending time & duration
        open_mesh_end = time.time()
        open_mesh_run = open_mesh_end - open_mesh_start
        print("Pre-Processing run time: %g seconds" % (open_mesh_run))

        print("Mesh Cells:", mesh.n_cells)

    def save_screenshot(self):
        ''' saves screenshot of current render window'''
        screenshot_path = 'screenshot/' + mesh_name + '.png'
        self.plotter.screenshot(screenshot_path)

    def skewed_centroid_check(self):
        ''' depending if the menu item is checked or not, the centroid is either skewed 
        with the 2nd moment of inertia or being the origin of the principal axes '''

        if self.skewed_centroid_action.isChecked():
            Vol_centroid = self.centroid(ext_surface)
        else:
            Vol_centroid = np.array([0,0,0])

        self.reset_plotter(Vol_centroid)

        return Vol_centroid

    def reset_plotter(self, Vol_centroid):
        """ clear plotter of mesh or interactive options """
        # clear plotter
        self.plotter.clear()
        
        # callback opened mesh
        self.plotter.add_mesh(mesh, show_edges = True, color="w", opacity=0.3)
        
        # show origin
        self.plotter.add_axes_at_origin(xlabel='X', ylabel='Y', zlabel='Z', line_width=6, labels_off=True)

        self.plotter.add_mesh(pv.PolyData(Vol_centroid), color='r', point_size=40, render_points_as_spheres=True)
        
    def centroid(self, mesh):
        """ find centroid volumetrically and indicate on graph """
        global V

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
        
        return Vol_centroid

    def show_max_cube(self):
        # check which centroid is used
        Vol_centroid = self.skewed_centroid_check()
        
        _, max_normal, intxn = self.max_cube_ray(int_surface, Vol_centroid, ext = True)
        self.max_cuboid(int_surface, intxn, Vol_centroid, max_normal)
        
    def max_cube_ray(self, mesh, Vol_centroid, ext = False):
        """ add a maximally inscribed cube within the opened mesh (via ray tracing) """
        global r_len
        global face_center, max_cube_vol, max_cube, max_cuboid
        global max_cube_start, max_cube_end, max_cube_run
        global max_cube_V, max_cube_F

        # initiate variables
        max_cube = 0
        max_cuboid = 0

        # find mesh vertices
        V = np.array(mesh.points)

        # find the nearest possible cube vertex from top rays & mesh intersection
        top_vert = self.cube_center_ray(mesh, Vol_centroid, 'z')
        top = self.furthest_pt(top_vert, Vol_centroid)

        # find the nearest possible cube vertex from bottom rays & mesh intersection
        bottom_vert = self.cube_center_ray(mesh, Vol_centroid, '-z')
        bottom = self.furthest_pt(bottom_vert, Vol_centroid)

        # find the nearest possible cube vertex between the two
        if top[0] < bottom[0]:
            p = top[1]
            V = top[2]
        else:
            p = bottom[1]
            V = bottom[2]
        
        # set the furthest ray intersection of the mesh as starting vertex of the cube
        intxn = V[p,:]

        # create and show max cube
        max_cube_V, max_cube_F, max_cube_vol = self.create_cube(intxn, Vol_centroid, np.array([0,0,Vol_centroid[2]]))
        max_cube = pv.PolyData(max_cube_V, max_cube_F)
        self.plotter.add_mesh(max_cube, show_edges=True, line_width=5, color="orange", opacity = 0.8)

        # find & show max cube face centers
        cell_center = pv.PolyData(max_cube_V, max_cube_F).cell_centers()
        face_center = np.array(cell_center.points)
        # self.plotter.add_mesh(cell_center, color="r", point_size=8, render_points_as_spheres=True)

        # find max cube face normals
        max_normal = pv.PolyData(max_cube_V, max_cube_F).cell_normals

        # max cube volume
        if (ext == False):
            max_cube_vol = float(format(max_cube_vol, ".2f"))
            print("Cube Center Volume:", max_cube_vol, "in^3")

        return face_center, max_normal, intxn

    def cube_center_ray(self, mesh, start, dir):
        ''' from starting point shoot out n rays to find vertices of possible cubes '''
        global r_num, r_rot, r_dec, r_len

        # initialize variables
        r_num = 1
        r_rot = np.pi/2
        r_dec = -2*np.pi/r_num
        ray_size = np.zeros((4, 3))
        r_dir = ray_size
        r_dir_norm = ray_size
        r_end = ray_size
        r_int = []
        ori_r_int = []

        l_wid = 10
        pt_size = 25
        rays = [0] * 4
        ints = [0] * 4

        # set ray length
        r_len = np.sqrt((x_range/2)**2 + (y_range/2)**2 + (z_range/2)**2)
        
        # create rays by rotating the first, which creates the cube with xyz axes as its face normals
        for i in range(0, r_num):
            for j in range(0, 4):
                if (j == 0) and (dir == 'z'):
                    r_dir[0] = np.array([np.sqrt(2)/2 * np.cos(np.pi/4 + r_dec * i), np.sqrt(2)/2 * np.sin(np.pi/4 + r_dec * i), 0.5])
                    r_dir_norm[0] = r_dir[0] / np.linalg.norm(r_dir[0])
                    r_end[0] = start + r_dir_norm[0] * r_len
                    # set rotation matrix about 'z'
                    R = self.rot_axis(np.array([0,0,1]))
                elif (j == 0) and (dir == '-z'):
                    r_dir[0] = np.array([np.sqrt(2)/2 * np.cos(np.pi/4 + r_dec * i), np.sqrt(2)/2 * np.sin(np.pi/4 + r_dec * i), -0.5])
                    r_dir_norm[0] = r_dir[0] / np.linalg.norm(r_dir[0])
                    r_end[0] = start + r_dir_norm[0] * r_len
                    # set rotation matrix about '-z'
                    R = self.rot_axis(np.array([0,0,-1]))
                else:
                    r_end[j] = np.dot(R(j*r_rot), (r_end[0] - start).T).T
                    r_end[j] = r_end[j] + start

                # perform ray trace
                r_pts, _ = mesh.ray_trace(start, r_end[j])

                # show rays
                # rays[j] = self.plotter.add_mesh(pv.Line(Vol_centroid, r_end[j]), color='w', line_width=l_wid)
                # ints[j] = self.plotter.add_mesh(pv.PolyData(r_pts[0]), color='w', point_size=pt_size)

                # create an array of ray intersections
                r_int = np.append(r_int, r_pts[0])
            
            r_int = np.reshape(r_int, (4,3))
            _, ori_p, ori_V = self.nearest_pt(r_int, start)
            r_int = []
            ori_r_int = np.append(ori_r_int, ori_V[ori_p,:])

        ori_r_int = np.reshape(ori_r_int, (r_num,3))
        
        return ori_r_int

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
        global edge_length

        if (axis[0] == 0) and (axis[1] == 0) and (axis[2] == 0):
            axis[2] = 1
            vert_trans = np.array([0,0,0])
        elif (starting_pt[0] == 0) and (starting_pt[1] == 0) and (starting_pt[2] == 0):
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
        edge_length = dia_dis * np.sin(np.pi/4) * 2
        print("Center cube edge length:", edge_length)
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

    def max_cuboid(self, mesh, nearest_pt, Vol_centroid, max_normal):
        ''' extend max cube into maximally inscribed cuboid '''
        global face_center, max_cuboid, max_cuboid_vol
        # fix max_normals
        dir_check = (face_center - Vol_centroid) * 2 / edge_length
        x_check = np.abs(np.around(max_normal[0,0] - dir_check[0,0]))
        y_check = np.abs(np.around(max_normal[0,1] - dir_check[0,1]))
        z_check = np.abs(np.around(max_normal[0,2] - dir_check[0,2]))
        print(x_check)
        print(y_check)
        print(z_check)

        print(max_normal)
        print(dir_check)

        if (x_check == 2) or (y_check == 2) or (z_check == 2):
            max_normal = -max_normal
 
        # find the 3 out of 6 normal directions the max cube can be extended towards
        ext_dir = np.empty(shape=(3,3)) 
        main_dir = nearest_pt - Vol_centroid

        ind = 0
        for i in range(0, 6):
            if np.dot(main_dir, max_normal[i]) < 0:
                ext_dir[ind] = max_normal[i]
                ind += 1

        # extend faces by shooting a ray from the 4 vertices on each extendable face
        # in the direction of its face normal. Find the nearest intersection and
        # it would be the limit of extension for that face
        for i in range(0, 3):
            F_ind = np.where((np.around(max_normal) == np.around(ext_dir[i])).all(axis=1))
            F_ind = F_ind[0][0]
            np.reshape(max_cube_F, (6,5))
            faces = np.reshape(max_cube_F, (6,5))
            print(faces)
            V_ind = faces[F_ind, 1:5]
            print(V_ind)
            current_V = np.vstack([max_cube_V[V_ind[0]], max_cube_V[V_ind[1]], max_cube_V[V_ind[2]], max_cube_V[V_ind[3]]])
            print(current_V)
            ext_V = self.ext_ray(mesh, current_V, ext_dir[i])
            max_cube_V[V_ind] = ext_V

        # create & show extended max cube
        max_cuboid = pv.PolyData(max_cube_V, max_cube_F)
        self.plotter.add_mesh(max_cuboid, show_edges=True, line_width=5, color="y", opacity = 0.4)

        # find face centers of extended max cube
        cell_center = max_cuboid.cell_centers()
        face_center = np.array(cell_center.points)

        # find face normals of the extended max cube
        max_normal = max_cuboid.cell_normals

        # extended max cube volume
        max_cuboid_vol = float(format(max_cuboid.volume, ".2f"))
        print("Extended Max Cube Volume:", max_cuboid_vol)

    def ext_ray(self, mesh, current_V, ext_dir):
        ''' shoot rays from vertices of a cube face towards face normal & obtain intersections with mesh '''
        # initialize variables
        ext_end = current_V + ext_dir * np.ones((4,1)) * r_len
        ext_dis = np.zeros(4)
        ext_rays = [0] * 6 * r_num
        ext_ints = [0] * 6 * r_num

        # set raytracing parameters
        l_wid = 3
        pt_size = 10

        # perform ray tracing per extending face vertex
        for i in range(0,4):
            ext_int, _ = mesh.ray_trace(current_V[i], ext_end[i])
            ext_dis[i] = np.sqrt((ext_int[0][0] - current_V[i][0])**2 + (ext_int[0][1] - current_V[i][1])**2
                                 + (ext_int[0][2] - current_V[i][2])**2)

            # show rays
            # ext_rays[i] = self.plotter.add_mesh(pv.Line(current_V[i], ext_end[i]), color='w', line_width=l_wid)
            # ext_ints[i] = self.plotter.add_mesh(pv.PolyData(ext_int[0]), color='w', point_size=pt_size)

        # extend vertices by the shortest intersection distance
        ext_V = current_V + ext_dir * np.ones((4,1)) * min(ext_dis)
        
        return ext_V

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