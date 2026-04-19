import vtk
import numpy as np
from scipy.io import loadmat
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QSlider, QLabel, QHBoxLayout,QGroupBox,QPushButton
from PyQt6.QtCore import Qt,QTimer
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
import json

subjid=12
mat_file = f'atlas/zebraatlasdata_s{subjid}.mat'
# Define file paths for outputs
cell_xyz_npy = f'atlas/cell_xyz_s{subjid}.npy'
colormap_roinames_json = 'atlas/colormap_roinames.json'
faces_vertices_vtp = 'atlas/faces_vertices.vtp'
fibers_vtp = 'atlas/fibers.vtp'
cell_label_indices_npy=f'atlas/cell_label_indices_s{subjid}.npy'
cell_timeseries_npy=f'atlas/cell_timeseries_s{subjid}.npy'


def load_mat_file(filepath):
    """Load the MATLAB .mat file."""
    data = loadmat(filepath)
    return data

def save_as_npy(cell_xyz, output_path):
    """Save the cell_xyz as an npy file."""
    np.save(output_path, cell_xyz)

def save_as_json(colormap, roinames, output_path):
    """Save colormap and roinames to a JSON file."""
    with open(output_path, 'w') as json_file:
        json.dump({'colormap': colormap.tolist(), 'roinames': roinames}, json_file)

def generate_normals(polydata):
    """Generate normals for the given polydata."""
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polydata)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOn()
    normal_generator.Update()
    return normal_generator.GetOutput()

def adaptive_reduce_mesh(polydata, reduction_percentage):
    """Reduce the number of faces in the polydata adaptively using vtkDecimatePro."""
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(reduction_percentage)  # Percentage reduction (e.g., 0.2 = 20% reduction)
    decimate.PreserveTopologyOn()  # Keep topology intact
    decimate.Update()

    return decimate.GetOutput()

def save_faces_vertices_as_vtk(faces_vertices, output_path, names, colors, reduction_percentage=0.2):
    """Save multiple sets of faces and vertices to a VTK PolyData file with different colors and names using vtkMultiBlockDataSet."""
    
    # Create a MultiBlockDataSet to store multiple polydata objects
    multiblock = vtk.vtkMultiBlockDataSet()

    for f in range(len(faces_vertices)):
        print(f"{f+1}/{len(faces_vertices)}")
        vertices = faces_vertices[f]['vertices'][0]
        faces = faces_vertices[f]['faces'][0] - 1  # Convert to 0-based indexing

        # Create VTK Points
        points = vtk.vtkPoints()
        for vertex in vertices:
            points.InsertNextPoint(vertex[:3])

        # Create VTK Cells (Polygons)
        polygons = vtk.vtkCellArray()
        for face in faces:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(len(face))
            for i, idx in enumerate(face):
                polygon.GetPointIds().SetId(i, idx)
            polygons.InsertNextCell(polygon)

        # Create PolyData for each face-vertices pair
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polygons)

        # Adaptively reduce the mesh
        reduced_polydata = adaptive_reduce_mesh(polydata, reduction_percentage)

        # Generate normals for the reduced polydata
        polydata_with_normals = generate_normals(reduced_polydata)

        # Add each polydata to the MultiBlock dataset
        multiblock.SetBlock(f, polydata_with_normals)

        # Assign metadata (name) to each block
        multiblock.GetMetaData(f).Set(vtk.vtkCompositeDataSet.NAME(), names[f])

    # Write the MultiBlockDataSet to a VTK file
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(multiblock)
    writer.Write()

def save_fibers_to_vtk(fibers,output_path):
    """Save fibers, soma, and terminals to a VTK PolyData (.vtp) file, including the disp_soma, disp_terminal info, and line thickness."""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    colors = vtk.vtkUnsignedCharArray()  # Store RGB colors
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    # Array for line thickness (wi)
    thickness_array = vtk.vtkFloatArray()
    thickness_array.SetName("Thickness")

    # Create arrays to store soma and terminal information
    soma_terminal_points = vtk.vtkPoints()
    soma_terminal_array = vtk.vtkIntArray()
    soma_terminal_array.SetName("Soma Terminal")

    # Prepare sphere sources for soma and terminal representation
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(16)
    sphere.SetRadius(1)
    sphere.Update()

    sphere_half = vtk.vtkSphereSource()
    sphere_half.SetThetaResolution(16)
    sphere_half.SetPhiResolution(16)
    sphere_half.SetRadius(1 / 2)
    sphere_half.Update()

    point_id = 0
    for f, tr in enumerate(fibers):
        print(f"{f+1}/{len(fibers)}")
        st = tr[:, 6]
        ed = tr[:, 0]
        pos = tr[:, 2:5]
        ty = tr[:, 1]
        wi = tr[:, 5]  # Line thickness array

        # Generate random color and convert to uint8
        color = (np.random.rand(3) * 255).astype(np.uint8)  # Random color for each fiber
        
        fl = np.zeros(len(ed))

        for i in range(len(tr) - 1):
            if st[i] == -1:
                continue
            sti = np.where(ed == st[i])[0]
            if len(sti) > 0:
                fl[sti] = 1

            vs = pos[i]
            ve = pos[i + 1]

            # Add points to VTK points list
            points.InsertNextPoint(vs)
            points.InsertNextPoint(ve)

            # Create a line connecting the two points
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_id)
            line.GetPointIds().SetId(1, point_id + 1)
            lines.InsertNextCell(line)

            # Add color information for this fiber (converted to uint8)
            colors.InsertNextTypedTuple(color)
            colors.InsertNextTypedTuple(color)

            # Add line thickness
            thickness_array.InsertNextValue(wi[i])

            # Soma detection and insertion
            id_soma = np.where(ty == 1)[0]
            if len(id_soma) > 0:
                for i in id_soma:
                    sid = np.where(st == ed[i])[0]
                    tid = np.where(ty[sid] == 1)[0]
                    if len(tid) > 0:
                        cid = np.concatenate(([i], sid[tid]))
                        ty[sid[tid]] = -100
                        cpos = np.mean(pos[cid, :], axis=0)
                    else:
                        cpos = pos[i]

                    soma_terminal_points.InsertNextPoint(cpos)
                    soma_terminal_array.InsertNextValue(1)  # Mark it as soma

            # Terminal detection and insertion
            id_term = np.where(fl == 0)[0]
            if len(id_term) > 0:
                for i in id_term:
                    soma_terminal_points.InsertNextPoint(pos[i])
                    soma_terminal_array.InsertNextValue(2)  # Mark it as soma

            point_id += 2

    # Create PolyData for the fibers
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(colors)
    polydata.GetCellData().AddArray(thickness_array)  # Add line thickness to the cell data

    # Write PolyData to a .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()

    polydata1 = vtk.vtkPolyData()
    polydata1.SetPoints(soma_terminal_points)
    polydata1.GetCellData().AddArray(soma_terminal_array)  # Add line thickness to the cell data

    # Write PolyData to a .vtp file
    '''
    writer1 = vtk.vtkXMLPolyDataWriter()
    writer1.SetFileName(output_path1)
    writer1.SetInputData(polydata1)
    writer1.Write()
    ''' 

    print(f"Saved fibers with soma, terminal, and thickness info to {output_path}")


def save_normals_with_vertices(faces, vertices, normals, filename):
    data = {
        "faces": faces.tolist(),
        "vertices": vertices.tolist(),
        "normals": [list(normals.GetTuple(i)) for i in range(normals.GetNumberOfTuples())]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)

def calculate_normals(vertices, faces):
    # Create polydata object
    points = vtk.vtkPoints()
    for vertex in vertices:
        points.InsertNextPoint(vertex[:3])  # Assuming vertices are 3D points

    polygons = vtk.vtkCellArray()
    for face in faces:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(face))
        for i, idx in enumerate(face):
            polygon.GetPointIds().SetId(i, idx)
        polygons.InsertNextCell(polygon)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)

    # Generate normals
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polydata)
    normal_generator.ComputePointNormalsOn()  # Or ComputeCellNormalsOn(), depending on your needs
    normal_generator.Update()

    normals = normal_generator.GetOutput().GetPointData().GetNormals()
    return normals  # Return the normals for future use


class vtkZebraWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.cell_label_indices = []
        self.cell_xyz = []
        self.colormap = []
        self.faces_vertices = []
        self.roinames=[]
        self.fibers = []
        self.cell_activity=[]
        self.flag_fiber = False
        self.flag_cells= False
        self.flag_regions = False
        self.actors_cell=[]
        self.actors_fiber=[]
        self.actors_fiber_soma=[]
        self.actors_fiber_terminal=[]
        self.actors_region=[]

        self.fiber_options =  {
            'scale': 1,
            'soma_merge': 1,
            'disp_terminal': 1,
            'disp_soma': 1,
            'step': 50
        }
        self.mode3d = True
        self.actors = []  # Store actors for later manipulation

        # Set up the VTK rendering window
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # Set up the layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.vtkWidget)

        # Create a group box for the checkboxes
        self.display_groupbox = QGroupBox("Display Options")
        self.display_groupbox_layout = QVBoxLayout()

        # Add checkboxes for display flags (Fiber, Cells, Regions) inside the group box
        self.regions_checkbox = QCheckBox("Display Regions")
        self.regions_checkbox.setChecked(self.flag_regions)
        self.regions_checkbox.stateChanged.connect(self.render)
        self.display_groupbox_layout.addWidget(self.regions_checkbox)

        self.cells_checkbox = QCheckBox("Display Cells")
        self.cells_checkbox.setChecked(self.flag_cells)
        self.cells_checkbox.stateChanged.connect(self.render)
        self.display_groupbox_layout.addWidget(self.cells_checkbox)
        
        self.fibers_checkbox = QCheckBox("Display Fibers")
        self.fibers_checkbox.setChecked(self.flag_fiber)
        self.fibers_checkbox.stateChanged.connect(self.toggle_fiber_groupbox)
        self.fibers_checkbox.stateChanged.connect(self.render)
        self.display_groupbox_layout.addWidget(self.fibers_checkbox)

        # Set the layout of the group box
        self.display_groupbox.setLayout(self.display_groupbox_layout)

        # Add the group box to the main layout
        self.layout.addWidget(self.display_groupbox)

        # Add a slider for transparency
        self.roi_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_opacity_slider.setMinimum(0)
        self.roi_opacity_slider.setMaximum(100)
        self.roi_opacity_slider.setValue(50)  # Default to 50% transparency
        self.roi_opacity_slider.valueChanged.connect(self.update_roi_opacity)
        self.layout.addWidget(QLabel("ROI Opacity:"))
        self.layout.addWidget(self.roi_opacity_slider)

        self.cell_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.cell_opacity_slider.setMinimum(0)
        self.cell_opacity_slider.setMaximum(100)
        self.cell_opacity_slider.setValue(50)  # Default to 50% transparency
        self.cell_opacity_slider.valueChanged.connect(self.update_cell_opacity)
        self.layout.addWidget(QLabel("Cell Opacity:"))
        self.layout.addWidget(self.cell_opacity_slider)

        self.start_button = QPushButton("Start Real-Time")
        self.start_button.clicked.connect(self.start_real_time_updates)
        self.layout.addWidget(self.start_button)
  
        # Create a group box for the soma and terminal display checkboxes
        self.fiber_display_groupbox = QGroupBox("Fiber Display")
        self.fiber_display_groupbox_layout = QVBoxLayout()

        # Checkboxes for toggling soma and terminal display inside the group box
        self.fiber_soma_checkbox = QCheckBox("Display Soma")
        self.fiber_soma_checkbox.setChecked(self.fiber_options['disp_soma'])
        self.fiber_soma_checkbox.stateChanged.connect(self.render)
        self.fiber_display_groupbox_layout.addWidget(self.fiber_soma_checkbox)

        self.fiber_terminal_checkbox = QCheckBox("Display Terminal")
        self.fiber_terminal_checkbox.setChecked(self.fiber_options['disp_terminal'])
        self.fiber_terminal_checkbox.stateChanged.connect(self.render)
        self.fiber_display_groupbox_layout.addWidget(self.fiber_terminal_checkbox)

        # Set the layout of the group box
        self.fiber_display_groupbox.setLayout(self.fiber_display_groupbox_layout)

        # Add the group box to the main layout
        self.layout.addWidget(self.fiber_display_groupbox)

        self.fibers_checkbox.setChecked(True)
        self.regions_checkbox.setChecked(True)
        self.cells_checkbox.setChecked(True)

        self.load_models()


    def toggle_fiber_groupbox(self, state):
        self.fiber_display_groupbox.setEnabled(state)

    def add_roi_checkboxes(self):
        """
        Dynamically create checkboxes for each mesh (ROI).
        """
        self.roi_checkboxes = []
        for roi_name in self.roinames:
            checkbox = QCheckBox(roi_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.toggle_roi_mesh)
            self.layout.addWidget(checkbox)
            self.roi_checkboxes.append(checkbox)
    
    def toggle_roi_mesh(self):
        """
        Toggle the visibility of mesh based on ROI checkbox.
        """
        for i, checkbox in enumerate(self.roi_checkboxes):
            self.actors[i].SetVisibility(checkbox.isChecked())
        self.vtkWidget.GetRenderWindow().Render()

    def load_regions_from_vtk(self,filepath):
        """Load multiple sets of faces and vertices from a VTK MultiBlockDataSet (.vtm) file and create actors with different colors."""
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(filepath)
        reader.Update()

        multiblock = reader.GetOutput()


        # Iterate over each block (polydata) in the multiblock dataset
        for i in range(multiblock.GetNumberOfBlocks()):
            polydata = multiblock.GetBlock(i)

            # Create a mapper and actor for each polydata block
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Set color for each actor based on the provided colors
            color = self.colormap[i, :3]  # Take only the first 3 elements (RGB)
            actor.GetProperty().SetColor(color[0], color[1], color[2])

            # Retrieve the name from metadata
            name = multiblock.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME())
            print(f"Loaded Block {i}: {name}")

            # Add the actor to the list
            self.actors_region.append(actor)

            self.renderer.AddActor(actor)

    def save_models(self):
        print("Files do not exist. Converting the .mat file...")
        # Load the data from the .mat file
        data = load_mat_file(mat_file)
        
        if not os.path.exists(cell_label_indices_npy):  
            cell_label_indices = data['cell_label_indices'].flatten()
            save_as_npy(cell_label_indices, cell_label_indices_npy)

        if not os.path.exists(cell_xyz_npy):
            cell_xyz = data['cell_xyz']
            save_as_npy(cell_xyz, cell_xyz_npy)

        if not os.path.exists(cell_timeseries_npy):  
            Z = data['Z']
            save_as_npy(Z, cell_timeseries_npy)

        # Save colormap and roinames to JSON
        if not os.path.exists(colormap_roinames_json):
            colormap = data['colormap']
            roinames = [str(name[0]) for name in data['roinames'][0]]  # Convert roinames to a list of strings
            save_as_json(colormap, roinames, colormap_roinames_json)

        if not os.path.exists(faces_vertices_vtp):
            # Save faces_vertices to VTK, adaptively reducing by 20%
            faces_vertices = [rfv[0] for rfv in data['faces_vertices'][0]]  # Assuming faces_vertices is a list of dicts
            save_faces_vertices_as_vtk(faces_vertices, faces_vertices_vtp, roinames, self.colormap, reduction_percentage=0.2)

        # Save fibers to VTK, reducing by 20%
        if not os.path.exists(fibers_vtp):  
            fibers = data['fibers'][0]
            save_fibers_to_vtk(fibers, fibers_vtp, reduction_percentage=0.2)

        

    def load_models(self):  
        if not all(os.path.exists(path) for path in [cell_xyz_npy, cell_timeseries_npy, cell_label_indices_npy, colormap_roinames_json, faces_vertices_vtp, fibers_vtp]):
            self.save_models()

        with open(colormap_roinames_json, 'r') as json_file:
            colormap_roinames_data = json.load(json_file)
            colormap = np.array(colormap_roinames_data['colormap'])
            roinames = np.array(colormap_roinames_data['roinames'])
        self.colormap = colormap
        self.roinames = roinames
        
        print("All files already exist. Loading the existing files...")
        self.cell_xyz = np.load(cell_xyz_npy)
        self.cell_label_indices = np.load(cell_label_indices_npy)
        self.load_neuron_cells()
        print(f"Loaded {len(self.cell_label_indices)} cells")
        self.load_regions_from_vtk(faces_vertices_vtp) # Load regions from VTK file
        self.load_fibers_from_vtk(fibers_vtp, self.fiber_options)  # Load fibers from VTK file
        self.time_series= np.load(cell_timeseries_npy)
        self.compute_thresholds()
        self.render()

    def load_regions_mat(self, default_opacity=0.5):
        """
        Render surfaces from cell arrays of faces and vertices using VTK.
        faces_vertices is expected to be a list of dictionaries with 'faces' and 'vertices' keys.
        colormap provides the colors for each surface.
        """
        for i, surf in enumerate(self.faces_vertices):
            faces = surf['faces'][0] - 1  # Convert to 0-based index (if necessary)
            vertices = surf['vertices'][0]

            # Create VTK points and polygons
            points = vtk.vtkPoints()
            for vertex in vertices:
                points.InsertNextPoint(vertex[:3])  # Ensure each vertex is a 3-element sequence

            polygons = vtk.vtkCellArray()
            for face in faces:
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(len(face))
                for j in range(len(face)):
                    polygon.GetPointIds().SetId(j, face[j])
                polygons.InsertNextCell(polygon)

            # Create a polydata object
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polygons)

            # Calculate normals
            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(polydata)
            normal_generator.ComputePointNormalsOn()  # Generate normals at points
            normal_generator.ComputeCellNormalsOn()   # Optionally, you can compute cell normals too
            normal_generator.Update()

            # Get the output with calculated normals
            polydata_with_normals = normal_generator.GetOutput()

            # Create a mapper with polydata including normals
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata_with_normals)

            # Create an actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.colormap[i][:3])  # Apply RGB color (ignore alpha if present)
            actor.GetProperty().SetOpacity(default_opacity) 

            self.actors_region.append(actor)
            # Add the actor to the renderer
            self.renderer.AddActor(actor)


    def load_neuron_cells(self):
        max_label = int(np.max(self.cell_label_indices))
        self.actors_cell = []  # Store actors for each label for real-time manipulation

        # Plot each label
        for l in range(1, max_label + 1):
            id = np.where(self.cell_label_indices == l)[0]
            if len(id) == 0:
                continue

            points = vtk.vtkPoints()
            for idx in id:
                points.InsertNextPoint(self.cell_xyz[idx])

            # Create a polydata object
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)

            # Create a glyph3D object to render points
            glyph3D = vtk.vtkGlyph3D()
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(2.0)
            glyph3D.SetSourceConnection(sphere.GetOutputPort())
            glyph3D.SetInputData(polydata)
            glyph3D.ScalingOff()
            glyph3D.Update()

            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph3D.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            rgb_color = self.colormap[l - 1][:3]  # Take only the first 3 elements (RGB)
            actor.GetProperty().SetColor(rgb_color)  # Set color for each label
            actor.GetProperty().SetOpacity(0.2)  # Low opacity for inactive cells

            self.actors_cell.append(actor)

            self.renderer.AddActor(actor)

    def activity_neuron_cells(self):
        max_label = int(np.max(self.cell_label_indices))
        # Plot each label
        for l in range(1, max_label + 1):
            id = np.where(self.cell_label_indices == l)[0]
            if len(id) == 0:
                continue

            points = vtk.vtkPoints()
            for idx in id:
                points.InsertNextPoint(self.cell_xyz[idx])

            # Create a polydata object
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)

            # Create a glyph3D object to render points
            glyph3D = vtk.vtkGlyph3D()
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(2.0)
            glyph3D.SetSourceConnection(sphere.GetOutputPort())
            glyph3D.SetInputData(polydata)
            glyph3D.ScalingOff()
            glyph3D.Update()

            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph3D.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            rgb_color = self.colormap[l - 1][:3]  # Take only the first 3 elements (RGB)
            actor.GetProperty().SetColor(rgb_color)  # Set color for each label
            actor.GetProperty().SetOpacity(0.2) 

            self.actors_cell.append(actor)

            self.renderer.AddActor(actor)

    def load_fibers_from_vtk(self,filepath, options):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()

        polydata = reader.GetOutput()

        # Get Soma and Terminal information
        soma_array = polydata.GetPointData().GetArray("Soma")
        terminal_array = polydata.GetPointData().GetArray("Terminal")
        thickness_array = polydata.GetCellData().GetArray("Thickness")

        # Create a mapper and actor for the fiber lines
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Apply line thickness from the thickness array
        if thickness_array:
            actor.GetProperty().SetLineWidth(np.mean(thickness_array))
        self.actors_fiber.append(actor)
        self.renderer.AddActor(actor)

        if soma_array:
            for i in range(soma_array.GetNumberOfTuples()):
                if soma_array.GetValue(i) == 1:
                    sphere = vtk.vtkSphereSource()
                    sphere.SetRadius(0.5 * options['scale'])
                    sphere.Update()

                    sphere_mapper = vtk.vtkPolyDataMapper()
                    sphere_mapper.SetInputData(sphere.GetOutput())

                    sphere_actor = vtk.vtkActor()
                    sphere_actor.SetMapper(sphere_mapper)
                    sphere_actor.SetPosition(polydata.GetPoint(i))
                    sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red color for soma

                    self.actors_fiber_soma.append(sphere_actor)
                    self.renderer.AddActor(sphere_actor)
        else:
            self.actors_fiber_soma = []

        if terminal_array:
            for i in range(terminal_array.GetNumberOfTuples()):
                if terminal_array.GetValue(i) == 1:
                    sphere = vtk.vtkSphereSource()
                    sphere.SetRadius(0.5 * options['scale'])
                    sphere.Update()

                    sphere_mapper = vtk.vtkPolyDataMapper()
                    sphere_mapper.SetInputData(sphere.GetOutput())

                    terminal_actor = vtk.vtkActor()
                    terminal_actor.SetMapper(sphere_mapper)
                    terminal_actor.SetPosition(polydata.GetPoint(i))
                    terminal_actor.GetProperty().SetColor(0, 0, 1)  # Blue color for terminal

                    self.actors_fiber_terminal.append(terminal_actor)
                    self.renderer.AddActor(terminal_actor)
        else:
            self.actors_fiber_terminal = []



    def load_fibers(self, fibers, options):
        hls = []
        hsoms = []
        hterms = []
        step = options.get('step', 1)
        # Create a sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.SetRadius(options['scale'])
        sphere.Update()

        sphere_half = vtk.vtkSphereSource()
        sphere_half.SetThetaResolution(16)
        sphere_half.SetPhiResolution(16)
        sphere_half.SetRadius(options['scale'] / 2)
        sphere_half.Update()

        colors = np.random.rand(len(fibers), 3)

        for f, tr in enumerate(fibers):
            if f % step == 0:
                print(f, len(fibers))
            else:
                continue

            st = tr[:, 6]
            ed = tr[:, 0]
            ty = tr[:, 1]
            wi = tr[:, 5]
            pos = tr[:, 2:5]

            fl = np.zeros(len(ed))
            if not self.mode3d:
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()

            for i in range(len(tr)):
                if st[i] == -1:
                    continue
                sti = np.where(ed == st[i])[0]
                if len(sti) > 0:
                    fl[sti] = 1

                vs = pos[sti][0]
                ve = pos[i]

                if self.mode3d:
                    # Create a line
                    line = vtk.vtkLineSource()
                    line.SetPoint1(vs)
                    line.SetPoint2(ve)
                    line.Update()
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(line.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(colors[f])
                    actor.GetProperty().SetLineWidth(wi[i])
                    self.actors_fiber.append(actor)
                    self.renderer.AddActor(actor)
                    hls.append(actor)
                else:
                    points.InsertNextPoint(vs)
                    points.InsertNextPoint(ve)
                
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, 2*i)
                    line.GetPointIds().SetId(1, 2*i+1)
                    lines.InsertNextCell(line)

            if not self.mode3d:
                poly_data = vtk.vtkPolyData()
                poly_data.SetPoints(points)
                poly_data.SetLines(lines)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_data)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(colors[f])
                actor.GetProperty().SetLineWidth(wi[0]) #i
                self.actors_fiber.append(actor)
                self.renderer.AddActor(actor)

            if options['disp_soma']:
                id_soma = np.where(ty == 1)[0]
                if len(id_soma) > 0:
                    for i in id_soma:
                        if options['soma_merge']:
                            sid = np.where(st == ed[i])[0]
                            tid = np.where(ty[sid] == 1)[0]
                            if len(tid) > 0:
                                cid = np.concatenate(([i], sid[tid]))
                                ty[sid[tid]] = -100
                                cpos = np.mean(pos[cid, :], axis=0)
                            else:
                                cpos = pos[i]
                        else:
                            cpos = pos[i]

                        sphere_actor = vtk.vtkActor()
                        sphere_mapper = vtk.vtkPolyDataMapper()
                        sphere_mapper.SetInputData(sphere.GetOutput())
                        sphere_actor.SetMapper(sphere_mapper)
                        sphere_actor.SetPosition(cpos)
                        sphere_actor.GetProperty().SetColor(1, 0, 0)
                        sphere_actor.GetProperty().SetEdgeVisibility(0)
                        self.actors_fiber_soma.append(sphere_actor)
                        self.renderer.AddActor(sphere_actor)


            if options['disp_terminal']:
                id_term = np.where(fl == 0)[0]
                if len(id_term) > 0:
                    for i in id_term:
                        term_actor = vtk.vtkActor()
                        term_mapper = vtk.vtkPolyDataMapper()
                        term_mapper.SetInputData(sphere_half.GetOutput())
                        term_actor.SetMapper(term_mapper)
                        term_actor.SetPosition(pos[i])
                        term_actor.GetProperty().SetColor(0, 0, 1)
                        term_actor.GetProperty().SetEdgeVisibility(0)
                        self.actors_fiber_terminal.append(term_actor)
                        self.renderer.AddActor(term_actor)

        
    
    def render(self):
        self.renderer.RemoveAllViewProps()

        if self.regions_checkbox.isChecked():
            for actor in self.actors_region:
                self.renderer.AddActor(actor)

        if self.cells_checkbox.isChecked():
            for actor in self.actors_cell:
                self.renderer.AddActor(actor)

        if self.fibers_checkbox.isChecked():
            for actor in self.actors_fiber:
                self.renderer.AddActor(actor)
            if self.fiber_soma_checkbox.isChecked():
                for actor in self.actors_fiber_soma:
                    self.renderer.AddActor(actor)
            if self.fiber_terminal_checkbox.isChecked():
                for actor in self.actors_fiber_terminal:
                    self.renderer.AddActor(actor)
    
        self.actors = self.renderer.GetActors()

        #light = vtk.vtkLight()
        #self.renderer.AddLight(light)
        self.renderer.SetBackground(0.1, 0.2, 0.3)  # Set background color to white

        # Render the scene
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def compute_thresholds(self):
        """Compute the threshold for each cell as mean + 1 standard deviation."""
        self.thresholds = []
        for cell in range(self.time_series.shape[0]):
            mean = np.mean(self.time_series[cell])
            std_dev = np.std(self.time_series[cell])
            threshold = mean + std_dev
            self.thresholds.append(threshold)

    def update_neuron_cells(self, current_time):
        """Update the visibility and color of cells based on the time series and thresholds."""
        for l, actor in enumerate(self.actors_cell):
            # Get the current activity for the cell at time `current_time`
            cell_activity = self.time_series[l, current_time]

            # Check if the activity exceeds the threshold
            if cell_activity > self.thresholds[l]:
                # Cell is "on": set color to yellow and full opacity
                #actor.GetProperty().SetColor(1, 1, 0)  # Yellow
                actor.GetProperty().SetOpacity(1.0)  # Full opacity
            else:
                # Cell is "off": set opacity low and retain the original color
                original_color = self.colormap[l][:3]
                #actor.GetProperty().SetColor(original_color)  # Original color
                actor.GetProperty().SetOpacity(0.2)  # Low opacity for inactive cells

        # Render the updates
        self.vtkWidget.GetRenderWindow().Render()

    def start_real_time_updates(self):
        """Start real-time updates with a 0.2-second interval."""
        self.current_time = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time_step)
        self.timer.start(200)  # 200 ms = 0.2 seconds

    def update_time_step(self):
        """Update the cells and time step in real-time."""
        if self.current_time < self.time_series.shape[1]:  # Check if within time range
            self.update_neuron_cells(self.current_time)
            self.current_time += 1  # Move to the next time step
        else:
            self.timer.stop()  # Stop the timer if we've reached the end of the time series


    def update_opacity(self, value):
        opacity = value / 100.0  # Convert slider value to opacity (0.0 - 1.0)
        for i in range(self.actors.GetNumberOfItems()):
            actor = self.actors.GetItemAsObject(i)
            actor.GetProperty().SetOpacity(opacity)
        self.vtkWidget.GetRenderWindow().Render()

    def update_cell_opacity(self, value):
        opacity = value / 100.0  # Convert slider value to opacity (0.0 - 1.0)
        for actor in self.actors_cell:
            actor.GetProperty().SetOpacity(opacity)
        self.vtkWidget.GetRenderWindow().Render()

    def update_roi_opacity(self, value):
        opacity = value / 100.0  # Convert slider value to opacity (0.0 - 1.0)
        for actor in self.actors_region:
            actor.GetProperty().SetOpacity(opacity)
        self.vtkWidget.GetRenderWindow().Render()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Virtual Zebrafish Model")
        self.zebra_widget = vtkZebraWidget()
        self.setCentralWidget(self.zebra_widget)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
