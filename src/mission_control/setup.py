from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mission_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jordan',
    maintainer_email='jordanpalafoxs@gmail.com',
    description='YASMIN state machine for AMR warehouse robot missions',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'state_machine_node = mission_control.state_machine_node:main',
        ],
    },
)
