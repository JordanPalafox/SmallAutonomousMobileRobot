from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jordan',
    maintainer_email='jordanpalafoxs@gmail.com',
    description='A* + Bug1 + PID navigation for the Puzzlebot AMR',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'waypoint_recorder = navigation.waypoint_recorder:main',
            'nav_node = navigation.nav_node:main',
        ],
    },
)
