import logging
import transmon_fluxonium_sim._version


__version__ = transmon_fluxonium_sim._version.__version__



logger = logging.getLogger(__name__)
logger.info(f'Imported transmon_fluxonium_simversion: {__version__}')
