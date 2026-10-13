# 🦾 I2RT driver

**References:** \[[Installation](../../README.md#installation)\] \[[Robot configuration](../../deployment/README.md#configuration)\]

| Entry | Purpose |
| :---: | --- |
| [`scripts/minimum_gello.py`](scripts/minimum_gello.py) | RPC server launched by the deployment runner |
| [`examples/bilateral_control.py`](examples/bilateral_control.py) | Bilateral control constants |
| [`kinematics_mj.py`](i2rt/robots/kinematics_mj.py) | MuJoCo/SciPy inverse kinematics with optional Mink solver |
| [Deployment robot](../../deployment/devices/robot.py) | Leader/follower roles and camera-session integration |

The deployment runner owns RPC processes for the configured CAN channels. Use one
process per CAN interface.

**References:** \[[License](LICENSE)\]
