import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, 
    QCheckBox, QHBoxLayout, QSlider, QLabel, QDialogButtonBox
)
from PyQt6.QtCore import Qt
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class ZAnato(QWidget):
    def __init__(self, parent, width, height):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.resize(self.width, self.height)

        # Set up the VTK rendering widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add checkboxes and sliders for visualization control
        self.checkbox_layout = QHBoxLayout()

        self.roi_checkbox = QCheckBox("ROI")
        self.roi_checkbox.setChecked(True)  # Enable Polydata by default
        self.neuron_checkbox = QCheckBox("Neuron")
        self.neuron_checkbox.setChecked(True)  # Enable Neuron Data by default

        # Add transparency sliders for each modality (Right = Opaque, Left = Transparent)
        self.roi_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_slider.setRange(0, 100)  # 0% to 100% transparency
        self.roi_slider.setValue(0)  # Default: fully opaque

        self.neuron_slider = QSlider(Qt.Orientation.Horizontal)
        self.neuron_slider.setRange(0, 100)  # 0% to 100% transparency
        self.neuron_slider.setValue(0)  # Default: fully opaque

        # Add click mode checkbox
        self.click_checkbox = QCheckBox("Enable Clicking")

        # Add ROI list button box
        self.roi_button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.roi_button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Show ROIs")
        self.roi_button_box.clicked.connect(self.display_roi_information)

        # Add widgets to the layout
        self.checkbox_layout.addWidget(self.roi_checkbox)
        self.checkbox_layout.addWidget(QLabel("Transparency:"))
        self.checkbox_layout.addWidget(self.roi_slider)
        self.checkbox_layout.addWidget(self.neuron_checkbox)
        self.checkbox_layout.addWidget(QLabel("Transparency:"))
        self.checkbox_layout.addWidget(self.neuron_slider)
        self.checkbox_layout.addWidget(self.click_checkbox)
        self.checkbox_layout.addWidget(self.roi_button_box)
        self.layout.addLayout(self.checkbox_layout)
        self.layout.addWidget(self.vtkWidget)

        # Connect the checkboxes and sliders to the corresponding methods
        self.roi_checkbox.stateChanged.connect(self.toggle_roi_visibility)
        self.neuron_checkbox.stateChanged.connect(self.toggle_neuron_visibility)
        self.roi_slider.valueChanged.connect(self.adjust_roi_transparency)
        self.neuron_slider.valueChanged.connect(self.adjust_neuron_transparency)
        self.click_checkbox.stateChanged.connect(self.toggle_click_mode)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Set up VTK renderer and render window
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = self.vtkWidget.GetRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        # Text actor for displaying labels
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.GetTextProperty().SetFontSize(24)
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White color
        self.renderer.AddActor2D(self.text_actor)

        # Load polydata and neuron data
        vtk_filename = "atlas/rzb_fvs36b.vtk"
        self.roi = self.load_vtk_file(vtk_filename)

        # Load ROI data from CSV
        labelname = "atlas/MPIN-Atlas_brain_region_Combined_brain_regions_index.csv"
        self.roi_labels = pd.read_csv(labelname)

        # Create a lookup table for ROI names based on their scalar values
        self.roi_lookup = {row['Grey_level']: row['Brain_region'] for _, row in self.roi_labels.iterrows()}

        neuron_filename = "atlas/zebrafish_cell_xyz.npy"
        self.coordinates, self.labels = self.load_neuron_npy_data(neuron_filename)

        # Load polydata visualization
        self.roi_actor = self.load_polydata(self.roi)
        self.renderer.AddActor(self.roi_actor)

        # Load neuron data visualization
        self.neuron_actor = self.load_neuron_data(self.coordinates, self.labels)
        self.renderer.AddActor(self.neuron_actor)

        # Dictionary to store ROI text actors for click events
        self.roi_text_actors = {}

        # Add orientation marker (3D axis with cube) in the top-right corner
        self.add_orientation_marker()

        # Variables for rotation interaction
        self.is_rotating = False
        self.last_mouse_position = None

        self.visualize()
        self.show()

        # Enable picking to interact with the visualization
        self.clicking_enabled = False
        self.setup_picking()

        self.interactor.AddObserver("LeftButtonPressEvent", self.start_rotation)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.end_rotation)
        self.interactor.AddObserver("MouseMoveEvent", self.rotate_cube)

        self.interactor.Start()

    def start_rotation(self, obj, event):
        self.is_rotating = True
        self.last_mouse_position = self.interactor.GetEventPosition()

    def end_rotation(self, obj, event):
        self.is_rotating = False
        self.last_mouse_position = None

    def rotate_cube(self, obj, event):
        if not self.is_rotating:
            return

        current_mouse_position = self.interactor.GetEventPosition()
        if self.last_mouse_position is None:
            self.last_mouse_position = current_mouse_position
            return

        # Calculate mouse movement vector
        dx = current_mouse_position[0] - self.last_mouse_position[0]
        dy = current_mouse_position[1] - self.last_mouse_position[1]

        # Determine rotation angle based on mouse movement
        angle = np.sqrt(dx**2 + dy**2) * 0.1  # Scale the rotation speed

        # Calculate rotation axis based on mouse direction
        rotation_axis = [dy, dx, 0.0]  # Rotate around z-axis for simplicity

        # Apply the rotation to the cube actor
        transform = vtk.vtkTransform()
        transform.RotateWXYZ(angle, rotation_axis)
        self.cube.SetUserTransform(transform)

        self.last_mouse_position = current_mouse_position
        self.renderWindow.Render()

    def resizeEvent(self, event):
        self.width = event.size().width()
        self.height = event.size().height()
        super().resizeEvent(event)

    def toggle_roi_visibility(self):
        """
        Toggle the visibility of the polydata actor.
        """
        visibility = self.roi_checkbox.isChecked()
        self.roi_actor.SetVisibility(visibility)
        self.visualize()

    def toggle_neuron_visibility(self):
        """
        Toggle the visibility of the neuron data actor.
        """
        visibility = self.neuron_checkbox.isChecked()
        self.neuron_actor.SetVisibility(visibility)
        self.visualize()

    def adjust_roi_transparency(self):
        """
        Adjust the transparency of the polydata actor based on the slider value.
        """
        transparency = self.roi_slider.value() / 100.0  # Convert to 0.0 - 1.0
        self.roi_actor.GetProperty().SetOpacity(1.0 - transparency)  # Invert slider direction
        self.visualize()

    def adjust_neuron_transparency(self):
        """
        Adjust the transparency of the neuron data actor based on the slider value.
        """
        transparency = self.neuron_slider.value() / 100.0  # Convert to 0.0 - 1.0
        self.neuron_actor.GetProperty().SetOpacity(1.0 - transparency)  # Invert slider direction
        self.visualize()

    def toggle_click_mode(self):
        """
        Toggle the clicking mode on or off.
        Disable all camera interaction when clicking mode is enabled.
        """
        self.clicking_enabled = self.click_checkbox.isChecked()
        if self.clicking_enabled:
            # Disable camera interaction
            self.interactor.Disable()
        else:
            # Enable camera interaction
            self.interactor.Enable()

    def setup_picking(self):
        """
        Set up VTK picking to handle mouse clicks on actors.
        """
        self.picker = vtk.vtkCellPicker()
        self.interactor.SetPicker(self.picker)
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click)

    def on_left_click(self, obj, event):
        """
        Handle left-click events to interact with polydata and highlight selected cells.
        """
        if not self.clicking_enabled:
            return  # Ignore clicks if clicking mode is disabled

        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        actor = self.picker.GetActor()
        if actor == self.roi_actor:
            cell_id = self.picker.GetCellId()
            if cell_id >= 0:
                self.highlight_cell(cell_id)

        self.visualize()

    def highlight_cell(self, cell_id):
        """
        Highlight a specific cell in the polydata by changing its color, and draw the object name as text on the scene.
        """
        if not self.roi:
            return

        # Get the scalar value at the clicked cell
        scalar_value = self.roi.GetCellData().GetScalars().GetTuple1(cell_id)
        roi_name = self.roi_lookup.get(scalar_value, "Unknown ROI")

        # Highlight the cell by changing its color to red
        lut = self.roi_actor.GetMapper().GetLookupTable()
        lut.SetTableValue(cell_id, 1.0, 0.0, 0.0, 1.0)  # Red color
        self.roi_actor.GetMapper().Update()

        # Set the label text and position it in the scene
        obj_name = f"Cell ID: {cell_id}, ROI: {roi_name}"
        self.text_actor.SetInput(obj_name)

        # Get world coordinates of the picked point
        pos = self.picker.GetPickPosition()
        self.text_actor.SetPosition(pos[0], pos[1])

        self.visualize()

    def display_roi_information(self):
        """
        Display the ROI names and their colors directly on the view, using text actors with colored fonts.
        """
        # Clear previous ROI information from the renderer
        self.clear_roi_information()

        # Display ROI information
        for i, row in self.roi_labels.iterrows():
            roi_name = row['Brain_region']
            roi_color = self.create_color_for_roi(i)

            # Create text actor for each ROI name
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(roi_name)
            text_actor.GetTextProperty().SetColor(roi_color)  # Set font color to ROI color
            text_actor.GetTextProperty().SetFontSize(14)  # Smaller font size to fit more labels
            text_actor.SetPosition(10, 20 + i * 20)  # Adjust position

            # Add the text actor to the renderer and store it for click handling
            self.renderer.AddActor2D(text_actor)
            self.roi_text_actors[text_actor] = row['Grey_level']

        self.visualize()

        # Set up picking for ROI text actors
        self.setup_roi_label_picking()

    def setup_roi_label_picking(self):
        """
        Set up interaction for picking ROI labels. When clicked, the corresponding ROI mesh is highlighted.
        """
        def on_text_actor_click(obj, event):
            clicked_actor = self.picker.GetActor2D()
            if clicked_actor in self.roi_text_actors:
                roi_id = self.roi_text_actors[clicked_actor]
                self.highlight_roi_mesh(roi_id)

        self.picker.AddObserver("EndPickEvent", on_text_actor_click)

    def highlight_roi_mesh(self, roi_id):
        """
        Highlight the entire mesh corresponding to the clicked ROI label.
        """
        # Highlight all cells with the corresponding scalar value (ROI ID)
        num_cells = self.roi.GetNumberOfCells()
        lut = self.roi_actor.GetMapper().GetLookupTable()

        for cell_id in range(num_cells):
            scalar_value = self.roi.GetCellData().GetScalars().GetTuple1(cell_id)
            if scalar_value == roi_id:
                lut.SetTableValue(cell_id, 1.0, 0.0, 0.0, 1.0)  # Highlight in red
            else:
                lut.SetTableValue(cell_id, *lut.GetTableValue(cell_id))  # Keep original color

        self.roi_actor.GetMapper().Update()
        self.visualize()

    def clear_roi_information(self):
        """
        Remove all previously added ROI text actors from the renderer.
        """
        actors = self.renderer.GetActors2D()
        for actor in actors:
            if isinstance(actor, vtk.vtkTextActor):  # Only remove text actors
                self.renderer.RemoveActor(actor)

    def create_color_for_roi(self, index):
        """
        Generate a color for each ROI based on the index.
        """
        np.random.seed(index)
        return np.random.rand(3).tolist()  # Return an RGB color tuple

    def add_orientation_marker(self):
        """
        Add a 3D orientation marker (axis) with an interactive cube in the top-right corner.
        """
        # Create an annotated cube actor for the orientation marker
        self.cube = vtk.vtkAnnotatedCubeActor()
        self.cube.SetXPlusFaceText("R")
        self.cube.SetXMinusFaceText("L")
        self.cube.SetYPlusFaceText("A")
        self.cube.SetYMinusFaceText("P")
        self.cube.SetZPlusFaceText("S")
        self.cube.SetZMinusFaceText("I")
        self.cube.GetTextEdgesProperty().SetColor(1, 1, 1)
        self.cube.GetTextEdgesProperty().SetLineWidth(1)

        # Create axes that will be longer than the cube
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.5, 1.5, 1.5)  # Set the length of the axes

        # Combine the cube and axes using a vtkOrientationMarkerWidget
        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(self.cube)
        self.marker_widget.SetInteractor(self.interactor)
        self.marker_widget.SetViewport(0.75, 0.75, 1.0, 1.0)  # Fixed in top-right corner
        self.marker_widget.SetEnabled(True)
        self.marker_widget.InteractiveOff()  # Disable interaction with the orientation cube

        # Prevent the cube from moving with the camera
        self.marker_widget.SetOutlineColor(0, 0, 0)
        self.marker_widget.SetInteractor(self.interactor)
        self.marker_widget.EnabledOn()

        # Set up picking for cube face clicks
        self.interactor.AddObserver("EndPickEvent", self.on_cube_face_click)

    def on_cube_face_click(self, caller, event):
        """
        Adjust the camera view based on the face of the orientation cube that was clicked.
        """
        picked_prop = caller.GetPickedProp()
        if picked_prop and isinstance(picked_prop, vtk.vtkAnnotatedCubeActor):
            face_text = picked_prop.GetFaceText()  # Get the text of the clicked face
            self.adjust_camera_view(face_text)

    def adjust_camera_view(self, face_text):
        """
        Adjust the camera view based on the face of the cube that was clicked.
        """
        camera = self.renderer.GetActiveCamera()

        if face_text == "R":  # Right view
            camera.SetPosition(1, 0, 0)
        elif face_text == "L":  # Left view
            camera.SetPosition(-1, 0, 0)
        elif face_text == "A":  # Anterior view
            camera.SetPosition(0, 1, 0)
        elif face_text == "P":  # Posterior view
            camera.SetPosition(0, -1, 0)
        elif face_text == "S":  # Superior view
            camera.SetPosition(0, 0, 1)
        elif face_text == "I":  # Inferior view
            camera.SetPosition(0, 0, -1)

        camera.SetFocalPoint(0, 0, 0)  # Keep focus on the origin
        camera.SetViewUp(0, 0, 1)  # Set the view up direction
        self.renderer.ResetCamera()
        self.visualize()

    def load_polydata(self, polydata):
        scalar_range = polydata.GetScalarRange()
        num_surfaces = int(scalar_range[1] - scalar_range[0] + 1)

        # Ensure the lookup table has enough colors
        num_colors = max(num_surfaces, 256)  # At least 256 colors
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarRange(scalar_range[0], scalar_range[1])

        lut = self.create_color_lookup_table(num_colors)
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor

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
        return point_actor

    def create_color_lookup_table(self, num_colors):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(num_colors)
        lut.Build()

        for i in range(num_colors):
            color = (
                vtk.vtkMath.Random(0.0, 1.0),
                vtk.vtkMath.Random(0.0, 1.0),
                vtk.vtkMath.Random(0.0, 1.0),
            )
            lut.SetTableValue(i, *color, 0.5)

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
        """
        Render the current scene.
        """
        self.renderer.ResetCamera()
        self.renderWindow.Render()

    def load_vtk_file(self, filename):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        return polydata

    def load_neuron_npy_data(self, filename):
        data = np.load(filename)
        coordinates = data[:, :3]
        labels = data[:, 3].astype(int)
        coordinates[:, [0, 1]] = coordinates[:, [1, 0]]  # Swap x and y for correct orientation
        return coordinates, labels


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    z_anato_widget = ZAnato(window, 800, 600)
    window.setCentralWidget(z_anato_widget)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
