import vtk
import numpy as np
from scipy.io import loadmat
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QSlider, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

vtk.vtkObject.GlobalWarningDisplayOn()

class VTKStreamlineTree(QWidget):
    def __init__(self, fibs, options=None):
        super().__init__()
        
        self.fibs = fibs
        self.options = options if options is not None else {
            'scale': 1,
            'soma_merge': 1,
            'disp_terminal': 1,
            'disp_soma': 1
        }
        self.mode3d = True

        # Set up the VTK rendering window
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        
        # Set up layout and controls
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.layout.addWidget(self.vtkWidget)
        
        # Checkboxes for toggling soma and terminal display
        self.soma_checkbox = QCheckBox("Display Soma")
        self.soma_checkbox.setChecked(self.options['disp_soma'])
        self.soma_checkbox.stateChanged.connect(self.update_vtk)
        self.layout.addWidget(self.soma_checkbox)
        
        self.terminal_checkbox = QCheckBox("Display Terminal")
        self.terminal_checkbox.setChecked(self.options['disp_terminal'])
        self.terminal_checkbox.stateChanged.connect(self.update_vtk)
        self.layout.addWidget(self.terminal_checkbox)
        
        # Slider for adjusting the scale of spheres
        scale_label = QLabel("Scale:")
        self.layout.addWidget(scale_label)
        
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(10)
        self.scale_slider.setValue(int(self.options['scale'] * 10))
        self.scale_slider.valueChanged.connect(self.update_vtk)
        self.layout.addWidget(self.scale_slider)
        
        self.vtk_actors = []

        # Initial VTK rendering
        self.update_vtk()

    def update_vtk(self):
        self.renderer.RemoveAllViewProps()
        self.vtk_actors.clear()
        
        self.options['disp_soma'] = self.soma_checkbox.isChecked()
        self.options['disp_terminal'] = self.terminal_checkbox.isChecked()
        self.options['scale'] = self.scale_slider.value() / 10.0
        
        hls, hsoms, hterms = self.vtk_streamline_tree(self.fibs, self.options)
        self.vtk_actors.extend(hls)
        self.vtk_actors.extend(hsoms)
        self.vtk_actors.extend(hterms)
        
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def vtk_streamline_tree(self, fibs, options):
        hls = []
        hsoms = []
        hterms = []

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

        colors = np.random.rand(len(fibs), 3)

        for f, tr in enumerate(fibs):
            if f % 10 == 0:
                print(f, len(fibs))
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

                self.renderer.AddActor(actor)
                hls.append(actor)

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

                        self.renderer.AddActor(sphere_actor)
                        hsoms.append(sphere_actor)

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

                        self.renderer.AddActor(term_actor)
                        hterms.append(term_actor)

        light = vtk.vtkLight()
        self.renderer.AddLight(light)
        self.renderer.SetBackground(1, 1, 1)  # Set background color to white

        return hls, hsoms, hterms

class MainWindow(QMainWindow):
    def __init__(self, fibs):
        super().__init__()

        self.setWindowTitle("VTK Streamline Tree")
        self.streamline_tree_widget = VTKStreamlineTree(fibs)
        self.setCentralWidget(self.streamline_tree_widget)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    # Load fibs data from the .mat file
    data = loadmat('atlas/fibs_orig_new.mat')
    fibs = data['fibs'][0]  # Extract fibs from the loaded data
    
    window = MainWindow(fibs)
    window.show()

    sys.exit(app.exec())
