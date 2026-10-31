"""Process API (``POST /v1/process``) for GLiClass serving: ``classify``, ``boolean`` and ``score`` tasks.

The contract follows the Knowledgator unified analyze API (``NEW_API.md``), served
as ``process``, limited to the task types GLiClass can run. The modules do not need
Ray or a model, so they can be used in-process and tested on their own:

* ``render``       input -> one Markdown text
* ``labels``       label forms and the text the model sees
* ``request``      request validation into task specs, ``defaults`` resolution
* ``engine``       row planning, batching, selection rules, result assembly
* ``capabilities`` what the served checkpoint can run
"""

from .engine import ProcessEngine
from .errors import ERROR_STATUS, APIError
from .render import RenderedInput, render_input
from .capabilities import Capabilities, capabilities_from_config

__all__ = [
    "ERROR_STATUS",
    "APIError",
    "Capabilities",
    "ProcessEngine",
    "RenderedInput",
    "capabilities_from_config",
    "render_input",
]
