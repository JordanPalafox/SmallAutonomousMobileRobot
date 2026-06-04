import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    # Include the Flask web files inside the Python package so they are
    # importable at runtime via __file__-relative paths in app.py.
    package_data={
        package_name: [
            'web/templates/*.html',
            'web/static/*.js',
            'web/static/*.css',
        ],
    },
    data_files=[
        # ament index marker
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jordan',
    maintainer_email='jordanpalafoxs@gmail.com',
    description='Flask web dashboard for AMR robot: real-time telemetry, MJPEG video stream, and mission control',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'dashboard_node = dashboard.dashboard_node:main',
        ],
    },
)
