import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'voice_control'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        (os.path.join('share', package_name, 'models'),
            glob('models/*.npz')),
        (os.path.join('share', package_name, 'models', 'hmm_models'),
            glob('models/hmm_models/*.npz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Puzzlebot Team',
    maintainer_email='a01198515@tec.mx',
    description='HMM+MFCC voice recognition for AMR voice command interface',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'voice_node = voice_control.voice_node:main',
        ],
    },
)
