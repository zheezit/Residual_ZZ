def _get_version() -> str:
    from pathlib import Path

    import versioningit

    import transmon_fluxonium_sim

    transmon_fluxonium_sim_path = Path(transmon_fluxonium_sim.__file__).parent
    return versioningit.get_version(project_dir=transmon_fluxonium_sim_path.parent)


__version__ = _get_version()
