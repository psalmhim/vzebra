import numpy as np
import vtk


def load_vtk_file(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def load_neuron_data(filename):
    data = np.load(filename)
    coordinates = data[:, :3]
    labels = data[:, 3].astype(int)

    # Swap the first and second columns (x and y coordinates)
    coordinates[:, [0, 1]] = coordinates[:, [1, 0]]

    return coordinates, labels


def create_color_lookup_table(num_surfaces, active_indices):
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


def assign_random_colors_to_labels(labels):
    unique_labels = np.unique(labels)
    label_colors = {}

    for label in unique_labels:
        label_colors[label] = (
            vtk.vtkMath.Random(0.0, 1.0),
            vtk.vtkMath.Random(0.0, 1.0),
            vtk.vtkMath.Random(0.0, 1.0),
        )

    return label_colors


def visualize_neurons_with_surfaces(polydata, coordinates, labels):
    scalar_range = polydata.GetScalarRange()
    num_surfaces = int(scalar_range[1] - scalar_range[0] + 1)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(scalar_range[0], scalar_range[1])

    lut = create_color_lookup_table(num_surfaces, None)
    mapper.SetLookupTable(lut)
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

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

    label_colors = assign_random_colors_to_labels(labels)
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

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(point_actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.Render()
    render_window_interactor.Start()


def main():
    vtk_filename = "atlas/zb_fvs36b.vtk"
    polydata = load_vtk_file(vtk_filename)

    neuron_filename = "atlas/zebrafish_cell_xyz.npy"
    coordinates, labels = load_neuron_data(neuron_filename)

    visualize_neurons_with_surfaces(polydata, coordinates, labels)


if __name__ == "__main__":
    main()
