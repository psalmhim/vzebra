import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore

# 절대 경로 import (상대 import 제거)
from zebra_v55.world.world_env import WorldEnv
from zebra_v55.world.world_renderer import WorldRenderer
from zebra_v55.agents.zebrafish_agent import create_agents


class SimulationController(QtWidgets.QMainWindow):
    def __init__(self, n_fish=1, device="cpu"):
        super().__init__()

        self.world = WorldEnv(
            xmin=-200, xmax=200,
            ymin=-150, ymax=150,
            n_food=15
        )

        self.agents = create_agents(
            world=self.world,
            n=n_fish,
            device=device,
            use_sparse_wiring=True,
            plasticity=False,
            dt=0.02
        )

        bodies = [ag.body for ag in self.agents]
        physics = [ag.physics for ag in self.agents]

        self.renderer = WorldRenderer(
            world=self.world,
            fish_list=self.agents,
            body_list=bodies,
            physics_list=physics,
            width=900,
            height=700,
            scale=2.0
        )

        self.setCentralWidget(self.renderer)
        self.setWindowTitle("Zebrafish Active Inference SNN — v55.1")
        self.resize(1000, 800)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(20)

    def update_sim(self):
        for ag in self.agents:
            ag.step(T_brain=1)

        self.renderer.update()


def main():
    app = QtWidgets.QApplication(sys.argv)
    ctrl = SimulationController(n_fish=1, device="cpu")
    ctrl.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
