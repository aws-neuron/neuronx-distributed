import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

from ._patched_functions import _apply_patches

logger.info("Applying patches to enable greedy sampling on XLA devices.")

_apply_patches()
