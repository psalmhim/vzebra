"""
Spiking tactile/somatosensory system: mechanoreceptor-based touch.

Zebrafish somatosensory system:
  - Rohon-Beard neurons: trunk/tail mechanosensory (larval, replaced by DRG in adults)
  - Trigeminal ganglion: head/face touch detection
  - Merkel cells: light touch on body surface
  - Distinct from lateral line (which detects water flow, not direct contact)

Free Energy Principle:
  Generative model predicts expected tactile input from motor commands
  (self-generated touch from swimming vs external contact).
  Unexpected touch (high PE) → startle/withdrawal.
  Self-generated touch (from tail oscillation) → attenuated response.

Architecture:
  6 RS neurons: 2 head/trigeminal, 2 trunk/Rohon-Beard, 2 tail
  + 3-channel TwoCompColumn for somatotopic prediction
"""
import torch
import torch.nn as nn
from zebrav2.spec import DEVICE
from zebrav2.brain.neurons import IzhikevichLayer
from zebrav2.brain.two_comp_column import TwoCompColumn


class SpikingTactile(nn.Module):
    def __init__(self, n_neurons=6, device=DEVICE):
        super().__init__()
        self.device = device
        self.n = n_neurons
        self.neurons = IzhikevichLayer(n_neurons, 'RS', device)
        self.neurons.i_tonic.fill_(-1.0)
        self.register_buffer('rate', torch.zeros(n_neurons, device=device))

        # FEP: 3 somatotopic channels (head, trunk, tail)
        self.pc = TwoCompColumn(n_channels=3, n_per_ch=4, substeps=8, device=device)

        # State
        self.head_touch = 0.0       # trigeminal activation
        self.trunk_touch = 0.0      # Rohon-Beard activation
        self.tail_touch = 0.0       # tail mechanoreceptors
        self.touch_intensity = 0.0  # overall intensity [0,1]
        self.contact_location = 'none'   # 'head', 'trunk', 'tail', 'none'
        self.startle_reflex = False      # head touch → escape
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0

        # Efference copy: suppress self-generated touch
        self._predicted_tail = 0.0
        self._swim_speed = 0.0

    @torch.no_grad()
    def forward(self, collision: bool = False, wall_proximity: float = 0.0,
                heading_to_wall: float = 0.0,
                conspecific_distance: float = 999.0,
                swim_speed: float = 0.0, tail_beat: float = 0.0,
                predator_distance: float = 999.0) -> dict:
        """
        collision: True if fish collided with obstacle
        wall_proximity: [0,1] nearness to boundary
        heading_to_wall: cos(angle) fish heading towards nearest wall [-1,1]
        conspecific_distance: px to nearest conspecific (shoaling touch)
        swim_speed: current swimming speed (for efference copy)
        tail_beat: current CPG tail beat amplitude [0,1]
        predator_distance: px to predator (for bite/attack contact)
        """
        self._swim_speed = swim_speed

        # --- Head touch (trigeminal) ---
        # Collision with heading toward wall → head contact
        head = 0.0
        if collision and heading_to_wall > 0.3:
            head = 0.8
        elif wall_proximity > 0.8 and heading_to_wall > 0.5:
            head = 0.4 * wall_proximity
        # Predator attack: frontal bite
        if predator_distance < 25:
            head += max(0, (25 - predator_distance) / 25) * 0.7
        head = min(1.0, head)
        self.head_touch = head

        # --- Trunk touch (Rohon-Beard) ---
        trunk = 0.0
        # Conspecific body contact during shoaling
        if conspecific_distance < 20:
            trunk = max(0, (20 - conspecific_distance) / 20) * 0.6
        # Lateral wall brush
        if wall_proximity > 0.6 and abs(heading_to_wall) < 0.3:
            trunk += 0.3 * wall_proximity
        # Predator lateral contact
        if predator_distance < 30:
            trunk += max(0, (30 - predator_distance) / 30) * 0.5
        trunk = min(1.0, trunk)
        self.trunk_touch = trunk

        # --- Tail touch ---
        # Self-generated: tail beat during swimming (efference copy attenuates)
        self_generated = tail_beat * 0.3
        # External: collision from behind, predator tail-chase
        external = 0.0
        if collision and heading_to_wall < -0.3:
            external = 0.5
        if predator_distance < 35 and heading_to_wall < 0:
            external += max(0, (35 - predator_distance) / 35) * 0.4
        # Efference copy: suppress self-generated component
        self._predicted_tail = self_generated
        tail = min(1.0, external + max(0, self_generated - self._predicted_tail * 0.8))
        self.tail_touch = tail

        # --- Overall touch intensity ---
        self.touch_intensity = max(head, trunk, tail)

        # Contact location
        if head >= trunk and head >= tail and head > 0.05:
            self.contact_location = 'head'
        elif trunk >= head and trunk >= tail and trunk > 0.05:
            self.contact_location = 'trunk'
        elif tail > 0.05:
            self.contact_location = 'tail'
        else:
            self.contact_location = 'none'

        # Startle reflex: unexpected head touch (trigeminal → Mauthner cell)
        self.startle_reflex = head > 0.5

        # --- Spiking encoding ---
        I = torch.zeros(self.n, device=self.device)
        I[0] = head * 15.0       # head trigeminal +
        I[1] = head * 8.0        # head trigeminal -
        I[2] = trunk * 12.0      # trunk Rohon-Beard +
        I[3] = trunk * 8.0       # trunk Rohon-Beard -
        I[4] = tail * 10.0       # tail +
        I[5] = tail * 8.0        # tail -

        for _ in range(10):
            self.neurons(I + torch.randn(self.n, device=self.device) * 0.3)
        self.rate.copy_(self.neurons.rate)

        # --- FEP prediction: self-touch vs external ---
        sensory = torch.tensor([float(head), float(trunk), float(tail)],
                               device=self.device, dtype=torch.float32)
        # Prediction: expect self-generated tail touch proportional to speed
        prediction = torch.tensor([0.0, 0.0, float(self._predicted_tail)],
                                  device=self.device, dtype=torch.float32)
        pc_out = self.pc(sensory_drive=sensory, prediction_drive=prediction)
        pe = pc_out['pe']
        self.prediction_error = float(pe.abs().mean())
        self.precision = float(pc_out['precision'].mean())
        self.free_energy = pc_out['free_energy']

        return {
            'head_touch': self.head_touch,
            'trunk_touch': self.trunk_touch,
            'tail_touch': self.tail_touch,
            'touch_intensity': self.touch_intensity,
            'contact_location': self.contact_location,
            'startle_reflex': self.startle_reflex,
            'prediction_error': self.prediction_error,
            'precision': self.precision,
            'free_energy': self.free_energy,
            'rate': float(self.rate.mean()),
        }

    def reset(self):
        self.neurons.reset()
        self.rate.zero_()
        self.pc.reset()
        self.head_touch = 0.0
        self.trunk_touch = 0.0
        self.tail_touch = 0.0
        self.touch_intensity = 0.0
        self.contact_location = 'none'
        self.startle_reflex = False
        self.prediction_error = 0.0
        self.precision = 1.0
        self.free_energy = 0.0
        self._predicted_tail = 0.0
        self._swim_speed = 0.0
