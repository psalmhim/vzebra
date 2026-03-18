import sys
import random
import vtk
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QToolBar, QToolButton, QDockWidget,QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt, QSize  # Ensure QSize is imported
from PyQt6.QtGui import QPixmap, QBrush, QIcon
import io
from qzeb_basic import *
from qzeb_qviz import *
from qzeb_zebra import *
from qzeb_seaworld import * 
from qzeb_zanato import *
from qzeb_physio import *
from qzeb_monitor import *

class StretchingGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        #self.setRenderHint(QPainter.RenderHint.Antialiasing)
        #self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        #self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def resizeEvent(self, event):
        """
        Override the resize event to scale the scene to fit the view.
        """
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.IgnoreAspectRatio)


WIDTH, HEIGHT = 1600, 600
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ocean Simulator with PyQt6")
        self.resize(1600, 900)  # Set the initial size but make it resizable

        # Central widget for layout management
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Main layout (vertical) to hold all content
        main_layout = QVBoxLayout(central_widget)

        # Top layout: Contains ocean_view on the left and physio/zanato on the right
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Ocean view setup
       
        self.ocean_scene = SeaWorld(1000, 600)  # Adjusted initial scene size
        self.ocean_view = StretchingGraphicsView(self.ocean_scene)
        self.ocean_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Dynamic sizing
        top_layout.addWidget(self.ocean_view, 2)  # Stretch factor of 2

        # Load sea plants and planktons
        self.ocean_scene.load_sea_plants()
        self.ocean_scene.load_planktons()
        self.ocean_scene.add_zebrafishes()

        # Right side of top_layout: ZPhysio and ZAnato stacked vertically
        right_top_layout = QVBoxLayout()
        top_layout.addLayout(right_top_layout, 1)  # Stretch factor of 1

        # ZPhysio view setup
        
        num_neurons = 50
        num_groups = 6  # Number of groups, each with a different color
        self.zphysio_scene = ZPhysio(500, 300, num_neurons=num_neurons, duration=1000, num_groups=num_groups)
        self.zphysio_view = StretchingGraphicsView(self.zphysio_scene)
        self.zphysio_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Dynamic sizing
        right_top_layout.addWidget(self.zphysio_view)

        # ZAnato view setup with VTK
        self.zanato_view = ZAnato(self,500,300)
        self.zanato_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Dynamic sizing
        right_top_layout.addWidget(self.zanato_view)

        # Bottom layout: terminal_log and terminal_debug side by side
        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)

        # terminal_log
        self.terminal_log = QTextEdit(self)
        self.terminal_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom_layout.addWidget(self.terminal_log)

        # terminal_debug
        self.terminal_debug = InteractiveTerminal(self)
        self.terminal_debug.locals['ocean'] = self.ocean_scene
        self.terminal_debug.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bottom_layout.addWidget(self.terminal_debug)

        # Redirect print statements to the terminal
        self.output_stream = OutputStream(self.terminal_log)

        # Timer for updating the scenes
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scenes)
        self.timer.start(16)  # ~60 FPS

    def add_toolbar(self, layout, title, view_widget):
        toolbar = QToolBar(title, self)
        toolbar.setIconSize(QSize(16, 16))
        
        float_button = QToolButton(self)
        float_button.setIcon(QIcon.fromTheme("window-new"))  # Use a floating window icon
        float_button.clicked.connect(lambda: self.float_view(view_widget))
        toolbar.addWidget(float_button)
        
        hide_button = QToolButton(self)
        hide_button.setIcon(QIcon.fromTheme("window-close"))  # Use a close window icon
        hide_button.clicked.connect(lambda: self.hide_view(view_widget))
        toolbar.addWidget(hide_button)
        
        layout.addWidget(toolbar)
        layout.addWidget(view_widget)

    def float_view(self, view_widget):
        dock = QDockWidget(view_widget.windowTitle(), self)
        dock.setWidget(view_widget)
        dock.setFloating(True)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def hide_view(self, view_widget):
        view_widget.hide()

    def update_scenes(self):
        self.ocean_scene.update()
        self.zphysio_scene.update()
        self.zanato_view.update()

    def resizeEvent(self, event):
        zphysio_view_size = self.zphysio_view.size()
        self.zphysio_scene.setSceneRect(0, 0, zphysio_view_size.width(), zphysio_view_size.height())
        ocean_view_size = self.ocean_view.size()
        self.ocean_scene.resize_event(ocean_view_size.width(), ocean_view_size.height())
        self.ocean_scene.setSceneRect(0, 0, ocean_view_size.width(), ocean_view_size.height())
        super().resizeEvent(event)


class OutputStream(io.StringIO):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def write(self, string):
        self.text_edit.insertPlainText(string)
        self.text_edit.ensureCursorVisible()

    def flush(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    # Redirect standard output to the QTextEdit terminal
    sys.stdout = window.output_stream
    sys.stderr = window.output_stream

    window.show()
    sys.exit(app.exec())
