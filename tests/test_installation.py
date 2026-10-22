"""Keep CUDA and native GPU wheels out of the normal installation."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement
import yaml


ROOT = Path(__file__).parents[1]


def test_default_dependencies_do_not_install_gpu_libraries():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(item) for item in project["dependencies"]]
    assert next(item for item in requirements if item.name == "jax").extras == set()
    assert not any(item.name.startswith(("jaxmg", "nvidia", "jax-cuda")) for item in requirements)
    hpc = [Requirement(item) for item in project["optional-dependencies"]["hpc"]]
    assert any(item.name == "jax" and "cuda12" in item.extras for item in hpc)
    for architecture in ("x86_64", "aarch64"):
        wheels = [item for item in hpc if item.name == "jaxmg" and item.marker.evaluate(
            {"sys_platform": "linux", "platform_machine": architecture},
        )]
        assert len(wheels) == 1
        assert "linux_" + architecture in wheels[0].url


def test_conda_environments_include_native_multinest_and_correct_extra():
    for filename, install in (("environment.yml", "-e ."),
                              ("environment-hpc.yml", "-e .[hpc]")):
        environment = yaml.safe_load((ROOT / filename).read_text())
        assert environment["channels"] == ["conda-forge", "nodefaults"]
        dependencies = environment["dependencies"]
        assert "multinest" in dependencies
        assert "pymultinest" in dependencies
        assert "python=3.11" in dependencies
        assert {"pip": [install]} in dependencies
