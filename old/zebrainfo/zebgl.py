import numpy as np
import matplotlib.pyplot as plt
import vtk

# Load the data
data = np.load("zebrainfo/zebrafish_cell_xyz.npy")

# Extract positions and classes
positions = data[:, :3]
classes = data[:, 3].astype(int)

# Create a VTK points object
points = vtk.vtkPoints()
for pos in positions:
    points.InsertNextPoint(pos)

# Create a VTK PolyData object to store points
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)

# Create a color lookup table for the classes
lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(len(np.unique(classes)))
lut.Build()

# Create an array to store the color of each point
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(4)
colors.SetName("Colors")


# Example: Assuming classes is a numpy array or list of class indices
max_class_value = max(classes)

# Assign a color to each point based on its class, including the alpha value
for cls in classes:
    r, g, b, a = [int(c * 255) for c in plt.cm.jet(cls / max_class_value)]
    colors.InsertNextTuple([r, g, b, a])


# Add colors to the polydata
polydata.GetPointData().SetScalars(colors)

# Create a mapper and actor for visualization
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polydata)
mapper.SetScalarRange(0, len(np.unique(classes)))
mapper.SetLookupTable(lut)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Set up the renderer, window, and interactor
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1)  # White background

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Start the visualization
render_window.Render()
render_window_interactor.Start()
