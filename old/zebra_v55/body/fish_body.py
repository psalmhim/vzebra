import numpy as np

# ======================================================================
# Jointed zebrafish body model (geometry only)
# ======================================================================

class FishBody:
    """
    Handles the geometry of a jointed zebrafish body:
        - head polygon
        - 7 joints
        - per-joint thickness
    """

    def __init__(self):
        # Head polygon (simple diamond-like shape)
        self.head_pts = np.array([
            [  0,   6],
            [ -7,  -3],
            [  0,  -3],
            [  7,  -3],
            [  0,   6]
        ], dtype=float)

        # Joint Y positions along body axis
        self.jY = np.array([0, -15, -30, -45, -60, -72, -88], dtype=float)

        # Joint half-widths (symmetric thickness)
        self.jW = np.array([14, 8, 7, 5, 3, 2, 1], dtype=float)

        # Number of joints
        self.N = len(self.jY)

        # Curvature memory (updated by tail CPG)
        self.curv = np.zeros(self.N, dtype=float)


    # ==================================================================
    # Update curvature from tail CPG drive
    # ==================================================================
    def update_curvature(self, tail_amp, turn_force, dt=0.02):
        """
        tail_amp: oscillation amplitude (from CPG)
        turn_force: directional bias (motor left-right)
        """
        base_freq = 12.0   # Hz tail beat
        phase_speed = 2 * np.pi * base_freq * dt

        # traveling wave from head→tail
        for i in range(self.N):
            phase = phase_speed * (i + 1)
            osc = tail_amp * np.sin(phase)
            bias = -turn_force * (i / self.N)  # stronger bias at tail
            self.curv[i] = 0.8 * self.curv[i] + 0.2 * (osc + bias)


    # ==================================================================
    # Compute joint 2D positions
    # ==================================================================
    def compute_skeleton(self, x, y, heading):
        """
        Returns joint positions as array shape [N, 2]
        """

        pts = np.zeros((self.N, 2), dtype=float)
        rad = heading

        # For each joint: rotate (0, jY[i]) by heading
        c = np.cos(rad)
        s = np.sin(rad)

        for i in range(self.N):
            px = 0.0
            py = self.jY[i]

            # Apply curvature bending around Z axis
            ang = rad + self.curv[i] * 0.12
            c2 = np.cos(ang)
            s2 = np.sin(ang)

            # rotate local point
            gx = x + px * c2 - py * s2
            gy = y + px * s2 + py * c2

            pts[i] = [gx, gy]

        return pts


    # ==================================================================
    # Compute body polygon segments for rendering
    # ==================================================================
    def compute_body_polylines(self, skeleton):
        """
        Using joint positions and widths, returns a list of pairs:
            left_edge[], right_edge[]
        """

        left_edge = []
        right_edge = []

        for i in range(self.N):
            cx, cy = skeleton[i]
            rad = np.arctan2(
                skeleton[min(i+1, self.N-1),1] - cy,
                skeleton[min(i+1, self.N-1),0] - cx
            )

            # perpendicular
            nx = -np.sin(rad)
            ny =  np.cos(rad)

            w = self.jW[i] * 0.5
            lx = cx + nx * w
            ly = cy + ny * w
            rx = cx - nx * w
            ry = cy - ny * w

            left_edge.append([lx, ly])
            right_edge.append([rx, ry])

        return np.array(left_edge), np.array(right_edge)

