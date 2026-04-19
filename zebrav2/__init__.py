"""
vzebra — Virtual Zebrafish: a downloadable whole-brain simulation.

Quick start:
    from zebrav2 import VirtualZebrafish, BrainConfig, WorldConfig, BodyConfig

    fish = VirtualZebrafish.load('pretrained')
    fish.lesion('habenula')
    results = fish.run(steps=1000)
"""
__version__ = '2.0.0'
