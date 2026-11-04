"""Task registrations (importing this package registers the environments with gymnasium)."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])
