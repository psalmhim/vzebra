import numpy as np
from .fish import ZebrafishLarva

class Shoal:
    def __init__(self, N=12):
        self.N = N
        self.fish_list = [ZebrafishLarva() for _ in range(N)]

        # Improved Couzin zones
        self.ZOR = 25     # repulsion (bigger to avoid overlap)
        self.ZOA = 60     # alignment
        self.ZOAT = 120   # attraction

        # force weights
        self.w_sep = 3.0
        self.w_align = 0.7
        self.w_coh = 0.45

        # steering gain
        self.steering_strength = 0.03

        # ----------------------------
        # INITIAL POSITION FIX ★★★★★
        # 모든 물고기를 화면 중앙부에 흩뿌리기
        # ----------------------------
        for f in self.fish_list:
            f.x = np.random.uniform(350, 850)
            f.y = np.random.uniform(350, 850)
            f.heading = np.random.uniform(-np.pi, np.pi)

    def step(self):
        pos = np.array([[f.x, f.y] for f in self.fish_list])
        heading = np.array([f.heading for f in self.fish_list])

        for i, f in enumerate(self.fish_list):
            p = pos[i]

            sep = np.zeros(2)
            align_list = []
            coh = np.zeros(2)

            for j in range(self.N):
                if i == j:
                    continue

                p2 = pos[j]
                dvec = p2 - p
                dist = np.linalg.norm(dvec)
                if dist < 1e-6:
                    continue

                # Repulsion
                if dist < self.ZOR:
                    sep -= dvec / (dist**2)

                # Alignment
                elif dist < self.ZOA:
                    align_list.append(heading[j])

                # Cohesion
                elif dist < self.ZOAT:
                    coh += dvec / dist

            # -------- compute steering --------
            # separation direction
            if np.linalg.norm(sep) > 1e-5:
                sep_force = np.arctan2(sep[1], sep[0])
            else:
                sep_force = 0.0

            # alignment direction
            if align_list:
                avg_h = np.mean(align_list)
                align_force = np.arctan2(
                    np.sin(avg_h - f.heading),
                    np.cos(avg_h - f.heading)
                )
            else:
                align_force = 0.0

            # cohesion direction
            if np.linalg.norm(coh) > 1e-5:
                coh_force = np.arctan2(coh[1], coh[0])
            else:
                coh_force = 0.0

            # final steering
            steer = (
                self.w_sep * sep_force +
                self.w_align * align_force +
                self.w_coh * coh_force
            )

            f.heading += self.steering_strength * steer
            f.update()
