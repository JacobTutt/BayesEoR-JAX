"""Keep GPU dependencies optional and install them from the package index."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import yaml


ROOT = Path(__file__).parents[1]


def test_default_dependencies_do_not_install_gpu_libraries():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(item) for item in project["dependencies"]]
    jax_requirement = next(item for item in requirements if item.name == "jax")
    assert jax_requirement.extras == set()
    assert "0.11.1" in jax_requirement.specifier
    assert "0.11.2" in jax_requirement.specifier
    assert "0.10.2" not in jax_requirement.specifier
    python_versions = SpecifierSet(project["requires-python"])
    assert "3.12" in python_versions and "3.11" not in python_versions
    assert not any(item.name.startswith(("jaxmg", "nvidia", "jax-cuda")) for item in requirements)
    hpc = [Requirement(item) for item in project["optional-dependencies"]["hpc"]]
    assert len(hpc) == 1
    assert hpc[0].name == "jaxmg" and hpc[0].extras == {"cuda12"}
    assert hpc[0].url is None and hpc[0].marker is None
    assert "1.4.1" in hpc[0].specifier and "0.0.7" not in hpc[0].specifier
    assert set(project["optional-dependencies"]) == {"hpc"}


def test_conda_environments_include_native_multinest_and_correct_extra():
    for filename, install, python in (("environment.yml", "-e .", "python=3.12"),
                                      ("environment-hpc.yml", "-e .[hpc]", "python=3.12")):
        environment = yaml.safe_load((ROOT / filename).read_text())
        assert environment["channels"] == ["conda-forge", "nodefaults"]
        dependencies = environment["dependencies"]
        assert "multinest" in dependencies
        assert "pymultinest" in dependencies
        assert python in dependencies
        assert {"pip": [install]} in dependencies
