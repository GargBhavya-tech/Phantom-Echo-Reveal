from setuptools import setup
import os
from glob import glob

package_name = 'phantom_echo_reveal'

setup(
    name=package_name,
    version='17.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')),
        (os.path.join('share', package_name, 'params'),
            glob('params/*.yaml')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PHANTOM Team',
    maintainer_email='team@phantom-echo.dev',
    description='PHANTOM-ECHO REVEAL v17 ROS2 simulation',
    license='MIT',
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': [
    #         'phantom_node = phantom_echo_reveal.phantom_node:main',
    #         'acoustic_node = phantom_echo_reveal.acoustic_node:main',
    #     ],
    # },
)
