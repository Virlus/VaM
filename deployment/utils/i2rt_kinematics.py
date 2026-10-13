"""Kinematics helpers for I2RT/YAM robots.

The preferred implementation reuses i2rt's Mink-based solver.  A MuJoCo and
SciPy fallback is provided because the deployment environment does not always
install Mink even though the YAM MuJoCo model is available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np


class I2RTKinematics(Protocol):
    """Minimal FK/IK interface used by :class:`devices.robot.I2RTRobot`."""

    def fk(self, q: np.ndarray, site_name: Optional[str] = None) -> np.ndarray: ...

    def ik(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[bool, np.ndarray]: ...


class MuJoCoI2RTKinematics:
    """Small numerical YAM FK/IK fallback using installed MuJoCo and SciPy."""

    def __init__(self, xml_path: str, site_name: str = "grasp_site") -> None:
        import mujoco

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._site_name = site_name
        self._joint_limits = np.asarray(self._model.jnt_range[:6], dtype=np.float64)

    def fk(self, q: np.ndarray, site_name: Optional[str] = None) -> np.ndarray:
        site_name = site_name or self._site_name
        site_id = self._mujoco.mj_name2id(
            self._model,
            self._mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise ValueError(f"Unknown I2RT site: {site_name}")

        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != 6:
            raise ValueError(f"I2RT arm FK expects 6 joints, got {q.size}")
        self._data.qpos[:6] = q
        self._mujoco.mj_forward(self._model, self._data)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(self._data.site_xmat[site_id]).reshape(3, 3)
        pose[:3, 3] = np.asarray(self._data.site_xpos[site_id])
        return pose

    def ik(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: Optional[np.ndarray] = None,
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-3,
        max_iters: int = 200,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[bool, np.ndarray]:
        del kwargs
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation as R

        target_pose = np.asarray(target_pose, dtype=np.float64)
        target_position = target_pose[:3, 3]
        target_rotation = R.from_matrix(target_pose[:3, :3])
        lower = self._joint_limits[:, 0]
        upper = self._joint_limits[:, 1]
        if init_q is None:
            init_q = (lower + upper) / 2.0
        initial = np.clip(np.asarray(init_q, dtype=np.float64).reshape(6), lower, upper)

        def residual(q: np.ndarray) -> np.ndarray:
            current_pose = self.fk(q, site_name)
            position_error = current_pose[:3, 3] - target_position
            current_rotation = R.from_matrix(current_pose[:3, :3])
            orientation_error = (target_rotation.inv() * current_rotation).as_rotvec()
            return np.concatenate([position_error, orientation_error])

        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=max_iters,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        solution = np.asarray(result.x, dtype=np.float64)
        achieved_pose = self.fk(solution, site_name)
        position_error = float(np.linalg.norm(achieved_pose[:3, 3] - target_position))
        achieved_rotation = R.from_matrix(achieved_pose[:3, :3])
        orientation_error = float(np.linalg.norm((target_rotation.inv() * achieved_rotation).as_rotvec()))
        success = bool(result.success) and position_error <= pos_threshold and orientation_error <= ori_threshold
        if verbose:
            print(
                "[MuJoCo I2RT IK] "
                f"success={success}, pos_err={position_error:.2e}, "
                f"ori_err={orientation_error:.2e}, q={solution}"
            )
        return success, solution


def create_i2rt_kinematics(
    site_name: str = "grasp_site",
) -> Tuple[I2RTKinematics, str]:
    """Create the preferred available I2RT kinematics implementation."""
    repo_root = Path(__file__).resolve().parents[1]
    i2rt_root = repo_root.parent / "packages" / "i2rt"
    i2rt_root_str = str(i2rt_root)
    if i2rt_root_str not in sys.path:
        sys.path.insert(0, i2rt_root_str)

    xml_path = i2rt_root / "i2rt" / "robot_models" / "yam" / "yam.xml"
    try:
        from i2rt.robots.kinematics_mj import Kinematics

        return Kinematics(str(xml_path), site_name), "i2rt_mink"
    except Exception as exc:
        solver = MuJoCoI2RTKinematics(str(xml_path), site_name)
        return solver, f"mujoco_scipy_fallback ({exc})"
