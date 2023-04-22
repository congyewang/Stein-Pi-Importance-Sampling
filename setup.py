from setuptools import setup


setup(
    name='stein_pi_thinning',
    version='0.1.0',
    description='Library of Stein Pi Thinning Algorithm',
    url='https://github.com/congyewang/Stein-Pi-Thinning',
    author='Congye Wang',
    license='MIT',
    packages=['stein_pi_thinning'],
    install_requires=['cvxopt', 'jax', 'numpy', 'quadprog', 'scipy', 'statsmodels', 'tqdm']
    )