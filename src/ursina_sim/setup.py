from setuptools import setup
import os
from glob import glob

package_name = 'ursina_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include config files in the install space
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
    ],
    install_requires=['setuptools', 'torch', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='ROS2 simulation of DPG+HER bots migrated from Ursina',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_node = ursina_sim.sim_node:main',
        ],
    },
)
