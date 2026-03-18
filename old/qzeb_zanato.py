import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ZAnato class representing the anatomy view using VTK for 3D visualization
class ZAnato(QWidget):
    def __init__(self, parent, width, height):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.resize(self.width, self.height)
        # Set up the VTK rendering widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.vtkWidget)
        self.setLayout(self.layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        # Set up VTK renderer and render window
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = self.vtkWidget.GetRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)

        # Initialize the VTK interactor, but do not start it
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()


        vtk_filename = "atlas/zb_fvs36b.vtk"
        polydata = self.load_vtk_file(vtk_filename)
        self.load_polydata(polydata)

        # Load the neuron data
        neuron_filename = "atlas/zebrafish_cell_xyz.npy"
        coordinates, labels = self.load_neuron_npy_data(neuron_filename)
        self.load_neuron_data(coordinates, labels)

        self.visualize()
        self.show()

        self.interactor.Start()

    def resizeEvent(self, event):
        self.width = event.size().width()
        self.height = event.size().height()
        # Optionally: Update VTK widget or other components here if necessary
        # The layout should automatically adjust the size of vtkWidget

        super().resizeEvent(event)


    def load_3d_model(self):
        # Example: Create a simple 3D sphere
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(5.0)
        sphere_source.SetPhiResolution(30)
        sphere_source.SetThetaResolution(30)

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add the actor to the renderer
        self.renderer.AddActor(actor)

        # Set the background color and camera angle
        self.renderer.SetBackground(0.1, 0.2, 0.4)  # Set background color
        self.renderer.ResetCamera()

    def load_polydata(self, polydata):
        scalar_range = polydata.GetScalarRange()
        num_surfaces = int(scalar_range[1] - scalar_range[0] + 1)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarRange(scalar_range[0], scalar_range[1])

        lut = self.create_color_lookup_table(num_surfaces, None)
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)

    def load_neuron_data(self, coordinates, labels):
        points = vtk.vtkPoints()
        for coord in coordinates:
            points.InsertNextPoint(coord)

        vertices = vtk.vtkCellArray()
        for i in range(len(coordinates)):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)

        point_polydata = vtk.vtkPolyData()
        point_polydata.SetPoints(points)
        point_polydata.SetVerts(vertices)

        label_colors = self.assign_random_colors_to_labels(labels)
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        for label in labels:
            r, g, b = [int(c * 255) for c in label_colors[label]]
            colors.InsertNextTuple3(r, g, b)

        point_polydata.GetPointData().SetScalars(colors)

        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputData(point_polydata)

        point_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)
        point_actor.GetProperty().SetPointSize(3)  # Set point size

        self.renderer.AddActor(point_actor)

    def create_color_lookup_table(self, num_surfaces, active_indices):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(num_surfaces)
        lut.Build()

        if not active_indices:
            active_indices = list(range(num_surfaces))

        for i in range(num_surfaces):
            if i in active_indices:
                color = (
                    vtk.vtkMath.Random(0.0, 1.0),
                    vtk.vtkMath.Random(0.0, 1.0),
                    vtk.vtkMath.Random(0.0, 1.0),
                )
                lut.SetTableValue(i, *color, 0.5)
            else:
                lut.SetTableValue(i, 1.0, 1.0, 1.0, 0.0)

        return lut

    def assign_random_colors_to_labels(self, labels):
        unique_labels = np.unique(labels)
        label_colors = {}

        for label in unique_labels:
            label_colors[label] = (
                vtk.vtkMath.Random(0.0, 1.0),
                vtk.vtkMath.Random(0.0, 1.0),
                vtk.vtkMath.Random(0.0, 1.0),
            )

        return label_colors

    def visualize(self):
        # Reset the camera to fit the data
        self.renderer.ResetCamera()

        # Render the scene
        self.vtkWidget.GetRenderWindow().Render()

    def update(self):
        # Instead of calling update(), call Render directly
        self.renderWindow.Render()

    def load_vtk_file(self,filename):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        return polydata

    def load_neuron_npy_data(self,filename):
        data = np.load(filename)
        coordinates = data[:, :3]
        labels = data[:, 3].astype(int)

        # Swap the first and second columns (x and y coordinates)
        coordinates[:, [0, 1]] = coordinates[:, [1, 0]]

        return coordinates, labels

