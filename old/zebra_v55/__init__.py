"""
zebra_v55
---------
Active Inference Zebrafish Simulation (v55.1)

이 패키지는 다음 모듈로 구성됩니다:

brain/
    - two_comp_layer.py
    - sparse_wiring.py
    - retina_sampling.py
    - zebrafish_snn_5k.py
    - decode_motor.py

body/
    - fish_body.py
    - fish_physics.py
    - fish_sensors.py
    - tail_cpg.py

world/
    - world_env.py
    - world_renderer.py

agents/
    - zebrafish_agent.py

run_sim.py : 실행 엔트리포인트
"""

# 편의를 위해 주요 클래스를 최상위 패키지에서 노출
from .agents.zebrafish_agent import ZebrafishAgent, create_agents
from .world.world_env import WorldEnv
from .world.world_renderer import WorldRenderer
from .brain.zebrafish_snn_5k import ZebrafishSNN_5k
