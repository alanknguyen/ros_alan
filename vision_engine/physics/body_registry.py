"""
vision_engine/physics/body_registry.py — Object Shape/Mass Registry for Physics

Maps rigid body names from OptiTrack to their physical properties
(shape, dimensions, mass) for PyBullet simulation.

Reads from the 'objects' section of scene_config.yaml.
Provides defaults for unknown objects (small sphere, 0.1 kg).
"""

from typing import Dict, Optional


# Default shape for unknown objects
DEFAULT_SHAPE = {
    "shape": "sphere",
    "radius": 0.02,
    "color": [0.5, 0.5, 0.5],
    "mass": 0.1,
}


class BodyRegistry:
    """
    Registry mapping rigid body names to physical properties.

    Parameters
    ----------
    config : dict
        The 'objects' section from scene_config.yaml.
    """

    def __init__(self, config: dict):
        self._registry: Dict[str, dict] = dict(config) if config else {}

    def get(self, name: str) -> dict:
        """
        Get shape info for a named rigid body.

        Returns the registered shape if known, or a default small sphere.

        Parameters
        ----------
        name : str
            Rigid body name (must match OptiTrack/Motive name).

        Returns
        -------
        info : dict
            Keys: shape, size/radius/height, color, mass.
        """
        return self._registry.get(name, DEFAULT_SHAPE)

    def has(self, name: str) -> bool:
        """Check if a rigid body is registered."""
        return name in self._registry

    def all_names(self):
        """Return all registered object names."""
        return list(self._registry.keys())
