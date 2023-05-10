from setuptools import setup

setup(
    name='stein_thinning',
    version='2.0.0',
    description='Optimally compress sampling algorithm outputs',
    license='MIT',
    packages=['stein_thinning'],
    install_requires=['numpy', 'scipy']
    )
