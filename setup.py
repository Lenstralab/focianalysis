import setuptools
import os

version = '2022.12.0'
smfish_version = '1b964f6415c574274f3e74b743f373d9201a4b0a'

with open("README.md", "r") as fh:
    long_description = fh.read()

# smfish depends on tllab_common, so that will be installed too
try:
    import smfish
    if smfish.__git_commit_hash__ == smfish_version:
        smfish = []
    else:
        smfish = [f'smfish[tllab_common]@git+https://github.com/Lenstralab/smFISH.git@{smfish_version}']
except (ImportError, AttributeError):
    smfish = [f'smfish[tllab_common]@git+https://github.com/Lenstralab/smFISH.git@{smfish_version}']

with open(os.path.join(os.path.dirname(__file__), 'focianalysis', '_version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")
    try:
        with open(os.path.join(os.path.dirname(__file__), '.git', 'HEAD')) as g:
            head = g.read().split(':')[1].strip()
        with open(os.path.join(os.path.dirname(__file__), '.git', head)) as h:
            f.write("__git_commit_hash__ = '{}'\n".format(h.read().rstrip('\n')))
    except Exception:
        f.write(f"__git_commit_hash__ = 'unknown'\n")


setuptools.setup(
    name="focianalysis",
    version=version,
    author="Lenstra lab NKI",
    author_email="t.lenstra@nki.nl",
    description="Foci analysis code for the Lenstra lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.rhpc.nki.nl/LenstraLab/focianalysis",
    packages=['focianalysis'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    tests_require=['pytest-xdist'],
    install_requires=['ipython', 'numpy', 'scipy', 'scikit_posthocs', 'tqdm', 'matplotlib', 'pandas', 'seaborn',
                      'pyyaml', 'tiffwrite'],
    extras_require={'smfish': smfish},
    entry_points={'console_scripts': ['foci_pipeline=focianalysis.foci_pipeline:main',
                                      'foci_figures=focianalysis.foci_plot_figures_combined:main']},
    package_data={'': ['*.yml']},
    include_package_data=True,
)
