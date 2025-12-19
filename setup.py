import os
from setuptools import setup, PEP420PackageFinder
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel
import os
import subprocess
from subprocess import CalledProcessError

class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True


class bdist_wheel_plat_only(bdist_wheel):

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        python, abi = 'py3', 'none'
        return python, abi, plat


def get_version(version_str):
    major, minor, patch = version_str.split(".")
    patch = os.getenv('VERSION_PATCH', patch)
    suffix = os.getenv('SUFFIX')
    if not suffix:
        try:
            suffix = f'{subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[0:8]}.dev'
        except CalledProcessError:
            suffix = 'dev'
    return f"{major}.{minor}.{patch}+{suffix}"


exec(open("src/neuronx_distributed/_version.py").read())
setup(
    name="neuronx-distributed",
    version=get_version(__version__), #noqa
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords="aws neuron",
    packages=PEP420PackageFinder.find(where="src"),
    package_dir={"": "src"},
    package_data={
        "": [
            "LICENSE",
        ]
    },
    entry_points={
        "console_scripts": ["nxd_convert_zero_checkpoints=neuronx_distributed.optimizer.convert_zero_checkpoints:main"],
    },
    install_requires=[
        "torch-neuronx",
        "torch-xla",
        "safetensors",
        "tenacity",
        "setuptools>=70.1"
    ],
    extras_require={
        "test": ["pytest", "pytest-forked", "pytest-cov", "pytest-xdist", "transformers==4.51.*"],
    },
    distclass=BinaryDistribution,
    cmdclass={
        'bdist_wheel': bdist_wheel_plat_only,
    },
)
