import vtk
import numpy as np
from scipy.io import loadmat
import os
import json
import gzip

subjid = 12
mat_file = f'atlas/zebraatlasdata_s{subjid}.mat.gz'
fibers_vtp = 'atlas/fibers.vtp'

def load_mat_file(filepath):
    """Load the MATLAB .mat file, handling .gz if necessary."""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = loadmat(f)
    else:
        data = loadmat(filepath)
    return data

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
    sphere.SetPhiResolution(16)
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

# Check if fibers file exists, if not, generate it
if not os.path.exists(fibers_vtp):
    print("Fibers file not found. Generating from .mat file...")
    data = load_mat_file(mat_file)
    fibers = data['fibers'][0]
    save_fibers_to_vtk(fibers, fibers_vtp)

# Custom interactor style for dragging synapses with mouse
class SynapseDragStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()
        self.Dragging = False
        self.LastPos = [0, 0]

    def OnLeftButtonDown(self):
        clickPos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)  # Increase pick tolerance
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
        if picker.GetActor() == synapse_actor:
            self.Dragging = True
            self.LastPos = clickPos
            self.GetInteractor().GetRenderWindow().SetDesiredUpdateRate(30)
        else:
            super().OnLeftButtonDown()

    def OnLeftButtonUp(self):
        self.Dragging = False
        super().OnLeftButtonUp()

    def OnMouseMove(self):
        if self.Dragging:
            currPos = self.GetInteractor().GetEventPosition()
            dx = currPos[0] - self.LastPos[0]
            dy = currPos[1] - self.LastPos[1]
            # Translate in x and y directions with smaller scale
            synapse_actor.AddPosition(dx * 0.1, dy * 0.1, 0)  # Reduced scale factor
            self.LastPos = currPos
            self.GetInteractor().GetRenderWindow().Render()
        else:
            super().OnMouseMove()

    def OnMouseWheelForward(self):
        synapse_actor.AddPosition(0, 0, 0.5)  # Smaller z movement
        self.GetInteractor().GetRenderWindow().Render()

    def OnMouseWheelBackward(self):
        synapse_actor.AddPosition(0, 0, -0.5)  # Smaller z movement
        self.GetInteractor().GetRenderWindow().Render()

# Load colormap and roinames
with open('atlas/colormap_roinames.json', 'r') as json_file:
    colormap_roinames_data = json.load(json_file)
    colormap = np.array(colormap_roinames_data['colormap'])
    roinames = np.array(colormap_roinames_data['roinames'])

# Load cell data
cell_xyz = np.load('atlas/cell_xyz_s12.npy')
cell_label_indices = np.load('atlas/cell_label_indices_s12.npy')

# Functions
def load_regions_from_vtk(filepath, renderer, colormap):
    """Load multiple sets of faces and vertices from a VTK MultiBlockDataSet (.vtm) file and create actors with different colors."""
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    multiblock = reader.GetOutput()
    actors = []

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
        actor.GetProperty().SetOpacity(0.3)

        renderer.AddActor(actor)
        actors.append(actor)
    return actors

def load_neuron_cells(cell_xyz, cell_label_indices, renderer, colormap):
    max_label = int(np.max(cell_label_indices))
    actors = []

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
        actor.GetProperty().SetOpacity(0.3)  # Low opacity for inactive cells

        renderer.AddActor(actor)
        actors.append(actor)
    return actors

def load_fibers_from_vtk(filepath, renderer):
    """Load fibers from VTK file and add to renderer."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    polydata = reader.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0/255.0, 250/255.0, 10/255.0)  # Dark green color
    actor.GetProperty().SetOpacity(1.0)

    renderer.AddActor(actor)
    return [actor]

# Load synapses
synap_xyz = np.load('atlas/Synapse_xyz_data.npy')

# Match centroids with cells
synap_centroid = synap_xyz.mean(axis=0)
cell_centroid = cell_xyz.mean(axis=0)
synap_xyz = synap_xyz - synap_centroid + cell_centroid

# Scale z to correct depth (assuming z is twice, so halve it)
synap_xyz[:, 2] *= 0.5

# Scale overall by 0.9
synap_xyz *= 0.9

# Move slightly towards head (anterior direction, assuming +x)
synap_xyz[:, 0] += 20  # Adjust this value as needed

# Scale y by 0.95
synap_xyz[:, 1] *= 0.9

# Create renderer and render window
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1200, 800)
render_window.SetWindowName("Synapse Visualization with Atlas")

# Create interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Load regions
region_actors = load_regions_from_vtk('atlas/faces_vertices.vtp', renderer, colormap)

# Load cells
cell_actors = load_neuron_cells(cell_xyz, cell_label_indices, renderer, colormap)

# Load fibers if file exists
fiber_actors = []
if os.path.exists('atlas/fibers.vtp'):
    fiber_actors = load_fibers_from_vtk('atlas/fibers.vtp', renderer)

# Create points for synapses
points = vtk.vtkPoints()
for syn in synap_xyz:
    points.InsertNextPoint(syn)

# Create polydata
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# Glyph for spheres
glyph3D = vtk.vtkGlyph3D()
sphere = vtk.vtkSphereSource()
sphere.SetRadius(1.0)
glyph3D.SetSourceConnection(sphere.GetOutputPort())
glyph3D.SetInputData(polydata)
glyph3D.ScalingOff()
glyph3D.Update()

# Mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(glyph3D.GetOutputPort())

synapse_actor = vtk.vtkActor()
synapse_actor.SetMapper(mapper)
synapse_actor.GetProperty().SetColor(0, 0.8, 0)  # Green
synapse_actor.GetProperty().SetOpacity(0.5)

renderer.AddActor(synapse_actor)
synapse_actors = [synapse_actor]

# Add text actor with instructions
text_actor = vtk.vtkTextActor()
text_actor.SetInput("Controls:\nR: Toggle Regions\nC: Toggle Cells\nS: Toggle Synapses\nF: Toggle Fiber Color\nI: Toggle Fiber Visibility\nA/D: Shift Synapses Left/Right\nW/Shift+S: Shift Synapses Forward/Back\nQ/E: Shift Synapses Up/Down\nZ: Reset Synapse Position\nMouse: Click and drag on synapses to move in X/Y\nWheel: Move synapses in Z\nQ: Quit")
text_actor.SetPosition(10, 10)
text_actor.GetTextProperty().SetFontSize(12)
text_actor.GetTextProperty().SetColor(0, 0, 0)
renderer.AddActor2D(text_actor)

# Fiber color toggle state
fiber_color_light = [False]  # Use list to avoid nonlocal

# Background
renderer.SetBackground(1, 1, 1)  # White
renderer.ResetCamera()

# Custom interactor style
style = SynapseDragStyle()
interactor.SetInteractorStyle(style)

# Override key press event
def key_press(obj, event):
    key = interactor.GetKeySym()
    dx, dy, dz = 0, 0, 0
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
    # Manual shift controls for synapses
    elif key == 'a' or key == 'A':  # Left
        dx = -1
    elif key == 'd' or key == 'D':  # Right
        dx = 1
    elif key == 'w' or key == 'W':  # Forward (anterior)
        dy = 1
    elif key == 'x' or key == 'X':  # Forward (anterior)
        dy = -1    
    elif key == 's' or key == 'S':  # Backward (posterior), but s is used for synapses, wait conflict
        if interactor.GetShiftKey():  # Shift+S for backward
            dy = -1
        else:
            for actor in synapse_actors:
                actor.SetVisibility(not actor.GetVisibility())
    elif key == 'q' or key == 'Q':  # Up (dorsal)
        dz = 1
    elif key == 'e' or key == 'E':  # Down (ventral)
        dz = -1
    elif key == 'z' or key == 'Z':  # Reset position
        synapse_actor.SetPosition(0, 0, 0)
    
    if dx or dy or dz:
        synapse_actor.AddPosition(dx, dy, dz)
    
    render_window.Render()

style.AddObserver("KeyPressEvent", key_press)

# Start
interactor.Initialize()
render_window.Render()
interactor.Start()