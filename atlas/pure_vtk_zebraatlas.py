import vtk
import numpy as np
from scipy.io import loadmat
import os
import json
import gzip

subjid = 12
mat_file = f'atlas/zebraatlasdata_s{subjid}.mat.gz'
# Define file paths for outputs
cell_xyz_npy = f'atlas/cell_xyz_s{subjid}.npy'
colormap_roinames_json = 'atlas/colormap_roinames.json'
faces_vertices_vtp = 'atlas/faces_vertices.vtp'
fibers_vtp = 'atlas/fibers_vtp'
cell_label_indices_npy = f'atlas/cell_label_indices_s{subjid}.npy'
cell_timeseries_npy = f'atlas/cell_timeseries_s{subjid}.npy'

def load_mat_file(filepath):
    """Load the MATLAB .mat file, handling .gz if necessary."""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = loadmat(f)
    else:
        data = loadmat(filepath)
    return data

def save_as_npy(cell_xyz, output_path):
    """Save the cell_xyz as an npy file."""
    np.save(output_path, cell_xyz)

def save_as_json(colormap, roinames, output_path):
    """Save colormap and roinames to a JSON file."""
    with open(output_path, 'w') as json_file:
        json.dump({'colormap': colormap.tolist(), 'roinames': roinames}, json_file)

def adaptive_reduce_mesh(polydata, reduction_percentage):
    """Reduce the number of faces in the polydata adaptively using vtkDecimatePro."""
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(reduction_percentage)  # Percentage reduction (e.g., 0.2 = 20% reduction)
    decimate.PreserveTopologyOn()  # Keep topology intact
    decimate.Update()
    return decimate.GetOutput()

def generate_normals(polydata):
    """Generate normals for the given polydata."""
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(polydata)
    normal_generator.ComputePointNormalsOn()
    normal_generator.ComputeCellNormalsOn()
    normal_generator.Update()
    return normal_generator.GetOutput()

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

def save_fibers_to_vtk(fibers, output_path):
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

        # Set color to black for better visibility on white background
        color = (0, 0, 0)  # Black color for all fibers

        fl = np.zeros(len(ed))

        for i in range(len(tr) - 1):
            if st[i] == -1:
                continue
            sti = np.where(ed == st[i])[0]
            if len(sti) > 0:
                fl[sti] = 1

        # Create lines for the fiber
        line_points = []
        for i in range(len(pos)):
            points.InsertNextPoint(pos[i])
            line_points.append(point_id)
            point_id += 1

        if len(line_points) > 1:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(line_points))
            for j, p in enumerate(line_points):
                line.GetPointIds().SetId(j, p)
            lines.InsertNextCell(line)

            # Set color for the line (per cell)
            colors.InsertNextTuple3(color[0]/255.0, color[1]/255.0, color[2]/255.0)
            thickness_array.InsertNextValue(np.mean(wi))

    # Create polydata for fibers
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetCellData().AddArray(colors)
    polydata.GetCellData().AddArray(thickness_array)

    # Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()

def save_models():
    print("Files do not exist. Converting the .mat file...")
    # Load the data from the .mat file
    data = load_mat_file(mat_file)

    if not os.path.exists(cell_label_indices_npy):
        cell_label_indices = data['cell_label_indices'].flatten()
        save_as_npy(cell_label_indices, cell_label_indices_npy)

    if not os.path.exists(cell_xyz_npy):
        cell_xyz = data['cell_xyz']
        save_as_npy(cell_xyz, cell_xyz_npy)


        faces_vertices = [rfv[0] for rfv in data['faces_vertices'][0]]  # Assuming faces_vertices is a list of dicts
        save_faces_vertices_as_vtk(faces_vertices, faces_vertices_vtp, roinames, colormap, reduction_percentage=0.2)

    # Save fibers to VTK, reducing by 20%
    if not os.path.exists(fibers_vtp):
        fibers = data['fibers'][0]
        save_fibers_to_vtk(fibers, fibers_vtp)

def load_regions_from_vtk(filepath, renderer, colormap, actors_list):
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
        color = colormap[i, :3]  # Take only the first 3 elements (RGB)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetOpacity(0.5)

        renderer.AddActor(actor)
        actors_list.append(actor)

def load_neuron_cells(cell_xyz, cell_label_indices, renderer, colormap, actors_list):
    max_label = int(np.max(cell_label_indices))
    actors_cell = []  # Store actors for each label

    # Plot each label
    for l in range(1, max_label + 1):
        id = np.where(cell_label_indices == l)[0]
        if len(id) == 0:
            continue

        points = vtk.vtkPoints()
        for idx in id:
            points.InsertNextPoint(cell_xyz[idx])

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

        rgb_color = colormap[l - 1][:3]  # Take only the first 3 elements (RGB)
        actor.GetProperty().SetColor(rgb_color)  # Set color for each label
        actor.GetProperty().SetOpacity(0.2)  # Low opacity for inactive cells

        renderer.AddActor(actor)
        actors_list.append(actor)

def load_synapses(renderer, actors_list):
    """Load synapses and add to renderer."""
    synap_xyz = np.load('atlas/Synapse_xyz_data.npy')
    
    points = vtk.vtkPoints()
    for syn in synap_xyz:
        points.InsertNextPoint(syn)

    # Create a polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Create a glyph3D object to render points
    glyph3D = vtk.vtkGlyph3D()
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)  # Smaller than cells
    glyph3D.SetSourceConnection(sphere.GetOutputPort())
    glyph3D.SetInputData(polydata)
    glyph3D.ScalingOff()
    glyph3D.Update()

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(1, 0, 0)  # Red for synapses
    actor.GetProperty().SetOpacity(0.5)  # Semi-transparent

    renderer.AddActor(actor)
    actors_list.append(actor)

def load_fibers_from_vtk(filepath, renderer, actors_list):
    """Load fibers from VTK file and add to renderer."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    polydata = reader.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0/255.0, 250/255.0, 10/255.0)  # Initial dark green color
    actor.GetProperty().SetOpacity(1.0)

    renderer.AddActor(actor)
    actors_list.append(actor)

def load_synapses(renderer, actors_list):
    """Load synapses and add to renderer."""
    synap_xyz = np.load('atlas/Synapse_xyz_data.npy')
    
    points = vtk.vtkPoints()
    for syn in synap_xyz:
        points.InsertNextPoint(syn)

    # Create a polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Create a glyph3D object to render points
    glyph3D = vtk.vtkGlyph3D()
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)  # Smaller than cells
    glyph3D.SetSourceConnection(sphere.GetOutputPort())
    glyph3D.SetInputData(polydata)
    glyph3D.ScalingOff()
    glyph3D.Update()

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(1, 0, 0)  # Red for synapses
    actor.GetProperty().SetOpacity(0.5)  # Semi-transparent

    renderer.AddActor(actor)
    actors_list.append(actor)
    max_label = int(np.max(cell_label_indices))
    actors_cell = []  # Store actors for each label

    # Plot each label
    for l in range(1, max_label + 1):
        id = np.where(cell_label_indices == l)[0]
        if len(id) == 0:
            continue

        points = vtk.vtkPoints()
        for idx in id:
            points.InsertNextPoint(cell_xyz[idx])

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

        rgb_color = colormap[l - 1][:3]  # Take only the first 3 elements (RGB)
        actor.GetProperty().SetColor(rgb_color)  # Set color for each label
        actor.GetProperty().SetOpacity(0.2)  # Low opacity for inactive cells

        renderer.AddActor(actor)
        actors_list.append(actor)

def main():
    # Check if files exist, if not, save them
    if not all(os.path.exists(path) for path in [cell_xyz_npy, cell_label_indices_npy, colormap_roinames_json, faces_vertices_vtp, fibers_vtp]):
        save_models()

    # Load data
    with open(colormap_roinames_json, 'r') as json_file:
        colormap_roinames_data = json.load(json_file)
        colormap = np.array(colormap_roinames_data['colormap'])
        roinames = np.array(colormap_roinames_data['roinames'])

    print("Loading existing files...")
    cell_xyz = np.load(cell_xyz_npy)
    cell_label_indices = np.load(cell_label_indices_npy)
    print(f"Loaded {len(cell_label_indices)} cells")

    # Create renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("Virtual Zebrafish Model - Pure VTK")

    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Load and add actors
    region_actors = []
    load_regions_from_vtk(faces_vertices_vtp, renderer, colormap, region_actors)
    cell_actors = []
    load_neuron_cells(cell_xyz, cell_label_indices, renderer, colormap, cell_actors)
    synapse_actors = []
    load_synapses(renderer, synapse_actors)
    fiber_actors = []
    load_fibers_from_vtk(fibers_vtp, renderer, fiber_actors)

    # Add text actor with instructions
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput("Controls:\nR: Toggle Regions\nC: Toggle Cells\nS: Toggle Synapses\nF: Toggle Fiber Color\nI: Toggle Fiber Visibility\nQ: Quit")
    text_actor.SetPosition(10, 10)
    text_actor.GetTextProperty().SetFontSize(14)
    text_actor.GetTextProperty().SetColor(0, 0, 0)
    renderer.AddActor2D(text_actor)

    # Fiber color toggle state
    fiber_color_light = [False]  # Use list to avoid nonlocal

    # Set background and reset camera
    renderer.SetBackground(1, 1, 1)  # White background
    renderer.ResetCamera()

    # Custom interactor style
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # Override key press event
    def key_press(obj, event):
        key = interactor.GetKeySym()
        if key == 'r' or key == 'R':
            for actor in region_actors:
                actor.SetVisibility(not actor.GetVisibility())
        elif key == 'c' or key == 'C':
            for actor in cell_actors:
                actor.SetVisibility(not actor.GetVisibility())
        elif key == 's' or key == 'S':
            for actor in synapse_actors:
                actor.SetVisibility(not actor.GetVisibility())
        elif key == 'f' or key == 'F':
            if fiber_color_light[0]:
                for actor in fiber_actors:
                    actor.GetProperty().SetColor(0/255.0, 90/255.0, 0/255.0)
                fiber_color_light[0] = False
            else:
                for actor in fiber_actors:
                    actor.GetProperty().SetColor(248/255.0, 214/255.0, 48/255.0)
                fiber_color_light[0] = True
        elif key == 'i' or key == 'I':
            for actor in fiber_actors:
                actor.SetVisibility(not actor.GetVisibility())
        render_window.Render()

    style.AddObserver("KeyPressEvent", key_press)

    # Start the interactor
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

if __name__ == "__main__":
    main()