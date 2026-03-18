import vtk

# Create a custom OpenGL render window
open_gl_render_window = vtk.vtkOpenGLRenderWindow()

# Set up a renderer and add it to the render window
renderer = vtk.vtkRenderer()
open_gl_render_window.AddRenderer(renderer)

# Render something simple
renderer.SetBackground(0.1, 0.2, 0.4)  # Set background color
open_gl_render_window.Render()  # Start the OpenGL rendering