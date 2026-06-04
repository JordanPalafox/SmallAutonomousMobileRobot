import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lifting'

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
    install_requires=['setuptools', 'spidev'],
    zip_safe=True,
    maintainer='Jordan',
    maintainer_email='jordanpalafoxs@gmail.com',
    description='FPGA lifter control via GPIO HAL for AMR forklift robot',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'lifting_node = lifting.lifting_node:main',
        ],
    },
)
