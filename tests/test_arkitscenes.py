"""Tests for the ARKitScenes adapter (src/edge/sensing/arkitscenes_loader.py).

These run WITHOUT any download: they build a faithful in-memory fixture and
assert the parsing + the critical world->camera -> camera->world inversion.
"""
import os
import tempfile
import numpy as np

from src.edge.sensing.arkitscenes_loader import (
    axis_angle_to_matrix,
    traj_line_to_camera_to_world,
    parse_pincam,
)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_axis_angle_identity():
    assert np.allclose(axis_angle_to_matrix([0, 0, 0]), np.eye(3))


def test_axis_angle_known_rotation():
    # 90 deg about +y
    R = axis_angle_to_matrix([0, np.pi / 2, 0])
    assert np.allclose(R, _rot_y(np.pi / 2), atol=1e-6)


def test_traj_inverts_world_to_camera():
    """The .traj stores world->camera; loader must return camera->world."""
    # ground-truth camera_to_world: position p, rotation R_wc
    p = np.array([0.3, 1.1, -0.2])
    R_wc = _rot_y(np.deg2rad(12.0))
    # world->camera (what .traj encodes)
    R_cw = R_wc.T
    t_cw = -R_cw @ p
    # axis-angle of R_cw
    theta = np.arccos(np.clip((np.trace(R_cw) - 1) / 2, -1, 1))
    v = np.array([R_cw[2, 1] - R_cw[1, 2],
                  R_cw[0, 2] - R_cw[2, 0],
                  R_cw[1, 0] - R_cw[0, 1]])
    rvec = v / (2 * np.sin(theta)) * theta
    line = f"1000.000 {rvec[0]} {rvec[1]} {rvec[2]} {t_cw[0]} {t_cw[1]} {t_cw[2]}"

    _, T_wc = traj_line_to_camera_to_world(line)
    # recovered camera position and rotation must match the originals
    assert np.allclose(T_wc[:3, 3], p, atol=1e-6), "camera position wrong (bad inversion)"
    assert np.allclose(T_wc[:3, :3], R_wc, atol=1e-6), "camera rotation wrong (bad inversion)"


def test_pincam_parse():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pincam")
        with open(p, "w") as f:
            f.write("256 192 211.9 211.9 127.5 95.5\n")
        intr = parse_pincam(p)
        assert intr["width"] == 256 and intr["height"] == 192
        assert abs(intr["fx"] - 211.9) < 1e-6 and abs(intr["cx"] - 127.5) < 1e-6
