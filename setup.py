from setuptools import setup


setup(
    name='stein_q_thinning',
    version='0.1.0',
    description='Library of Stein Q Thinning Algorithm',
    url='https://github.com/congyewang/Stein-Q-Thinning',
    author='Congye Wang',
    license='MIT',
    packages=['stein_q_thinning'],
    install_requires=['jax', 'numpy', 'scipy', 'statsmodels', 'tqdm']
    )