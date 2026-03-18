import vtk
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
from vtkmodules.vtkRenderingMetal import vtkMetalRenderWindow

# Create a renderer and a render window that uses Metal
renderer = vtkRenderer()

# Create a Metal-based render window
renderWindow = vtkMetalRenderWindow()
renderWindow.AddRenderer(renderer)

# Set up a render window interactor for interaction
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# Create a simple VTK object (e.g., a sphere)
sphere_source = vtk.vtkSphereSource()
sphere_source.SetRadius(5.0)

# Mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(sphere_source.GetOutputPort())

# Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Add the actor to the renderer
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.4)  # Set background color

# Render the scene
renderWindow.Render()

# Start interaction
interactor.Initialize()
interactor.Start()
