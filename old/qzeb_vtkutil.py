import vtk


def load_vtk_file(filename):
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get the output
    polydata = reader.GetOutput()
    return polydata


def convert_vtk_ascii_to_binary(input_filename, output_filename):
    # Step 1: Read the ASCII VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_filename)
    reader.Update()

    # Step 2: Get the output from the reader
    polydata = reader.GetOutput()

    # Step 3: Write the data to a new binary VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()  # Set file type to binary
    writer.Write()


def decimate_polydata_with_scalars(polydata, reduction=0.5):
    # Use vtkQuadricDecimation which can sometimes better preserve scalars
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(reduction)
    decimate.AttributeErrorMetricOn()  # Preserve scalar attributes
    decimate.Update()

    return decimate.GetOutput()


def create_color_lookup_table(num_surfaces, active_indices):
    # Create a lookup table to map surface IDs to colors
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(num_surfaces)
    lut.Build()

    # If active_indices is None or empty, display all surfaces
    if not active_indices:
        active_indices = list(range(num_surfaces))

    # Generate colors for each surface ID
    for i in range(num_surfaces):
        if i in active_indices:
            color = (
                vtk.vtkMath.Random(0.0, 1.0),
                vtk.vtkMath.Random(0.0, 1.0),
                vtk.vtkMath.Random(0.0, 1.0),
            )
            lut.SetTableValue(i, *color, 0.5)  # RGB + Alpha
        else:
            lut.SetTableValue(i, 1.0, 1.0, 1.0, 0.0)  # Transparent

    return lut


def visualize_polydata_with_colors(polydata, active_indices=None):
    # Get the range of the scalar data
    scalar_range = polydata.GetScalarRange()
    num_surfaces = int(scalar_range[1] - scalar_range[0] + 1)

    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(scalar_range[0], scalar_range[1])

    # Create and assign the color lookup table
    lut = create_color_lookup_table(num_surfaces, active_indices)
    mapper.SetLookupTable(lut)
    mapper.ScalarVisibilityOn()

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()


def binary_decimate():
    # Load the VTK file
    filename = "atlas/zb_fvs36_binary.vtk"
    polydata = load_vtk_file(filename)

    # Decimate the polydata
    decimated_polydata = decimate_polydata_with_scalars(polydata, reduction=0.5)

    # Save the decimated polydata
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("atlas/zb_fvs36_decimated.vtk")
    writer.SetInputData(decimated_polydata)
    writer.SetFileTypeToBinary()
    writer.Write()


def main():
    # Load the VTK file
    filename = "atlas/zb_fvs36b.vtk"  # Replace with your binary VTK file path
    polydata = load_vtk_file(filename)

    # List of active indices (ROIs to show)
    # If you want to show all surfaces, you can pass an empty list or None
    active_indices = None  # Example: [0, 1, 2, 5, 10] to show specific indices

    # Visualize the polydata with assigned colors based on active indices
    visualize_polydata_with_colors(polydata, active_indices)


if __name__ == "__main__":
    main()
