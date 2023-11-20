from setuptools import setup


setup(
    name='stein_pi_is',
    version='0.1.1',
    description='Library of Stein Pi Importance Sampling Algorithm',
    url='',
    license='MIT',
    packages=['stein_pi_is'],
    install_requires=['jax', 'numpy', 'proxsuite', 'qpsolvers', 'scipy', 'statsmodels', 'tqdm']
    )