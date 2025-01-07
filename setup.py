from setuptools import setup, find_packages

setup(
    name='mjzoo',
    version='0.1',
    author='Rafi Darmawan',
    author_email='drmwnnrafi@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mujoco',
        'glfw',
    ],
)
