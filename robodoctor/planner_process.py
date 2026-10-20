"""Bound native cuRobo calls without interrupting Isaac's CUDA context."""
import faulthandler
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback


def _worker(connection, config, log_path):
    # The launcher may have to kill a wedged simulator. Do not leave a CUDA
    # worker orphaned after its owner dies (Linux, our RoboDojo platform).
    import ctypes
    import signal
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)
    if parent == 1 or os.getppid() != parent:
        return
    with open(log_path, 'a', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        faulthandler.enable(log)
        try:
            from .cuda_runtime import preload_nvrtc
            preload_nvrtc()
            from env.planner_manager.curobo_planner import CuroboPlanner
            planner = CuroboPlanner(**config)
            connection.send(dict(ok=True, result=dict(ee_link=planner.ee_link)))
            while True:
                method, args, kwargs = connection.recv()
                started = time.monotonic()
                print(json.dumps(dict(event='planner_started', method=method)), flush=True)
                faulthandler.dump_traceback_later(30, repeat=True, file=log)
                try:
                    result = getattr(planner, method)(*args, **kwargs)
                    connection.send(dict(ok=True, result=result))
                except Exception as error:
                    traceback.print_exc()
                    connection.send(dict(ok=False, error=f'{type(error).__name__}: {error}'))
                finally:
                    faulthandler.cancel_dump_traceback_later()
                    print(json.dumps(dict(event='planner_finished', method=method,
                                          seconds=time.monotonic()-started)), flush=True)
        except EOFError:
            pass
        except BaseException as error:
            traceback.print_exc()
            try:
                connection.send(dict(ok=False, error=f'{type(error).__name__}: {error}'))
            except (OSError, EOFError):
                pass
        finally:
            connection.close()


class PlannerProcess:
    def __init__(self, config, log_path, *, timeout=120, startup_timeout=600):
        self.dt = config['dt']
        self.timeout = timeout
        self.log_path = str(log_path)
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, config, self.log_path), daemon=True)
        self.process.start()
        child.close()
        try:
            self.ee_link = self._receive(startup_timeout, 'initialization')['ee_link']
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout, method):
        if not self.connection.poll(timeout):
            self.close()
            raise RuntimeError(f'cuRobo {method} exceeded {timeout}s; no motion executed. '
                               f'Planner stopped; see {self.log_path}')
        try:
            response = self.connection.recv()
        except (EOFError, OSError) as error:
            self.close()
            raise RuntimeError(f'cuRobo worker exited during {method}; see {self.log_path}') from error
        if not response['ok']:
            self.close()
            raise RuntimeError(response['error'])
        return response['result']

    def _call(self, method, *args, **kwargs):
        if not self.process.is_alive():
            raise RuntimeError('cuRobo worker unavailable; no motion executed. Restart the run.')
        try:
            self.connection.send((method, args, kwargs))
            return self._receive(self.timeout, method)
        except (EOFError, OSError) as error:
            self.close()
            raise RuntimeError(f'cuRobo worker connection failed; see {self.log_path}') from error

    def solve_ik_to_joint(self, **kwargs):
        return self._call('solve_ik_to_joint', **kwargs)

    def plan_joint(self, *args):
        return self._call('plan_joint', *args)

    def close(self):
        self.connection.close()
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)
        self.process.join(0)


def install_target_planners(manager, output, owned):
    """Match RobotManager._setup_planner's config; keep support arms untouched."""
    Path(output).mkdir(parents=True, exist_ok=True)
    for robot in manager.robot_list:
        if robot.type != 'target' or robot.robot_type != 'arm':
            continue
        if robot.robot_name in manager.planner:
            continue
        config = dict(robot_origin_pose=[0., 0., 0., 1., 0., 0., 0.],
                      active_joints_name=list(robot.arm_joints_name), all_joints=list(robot.arm_joints_name),
                      dt=manager.dt, yml_path=str(robot.curobo_yml_path),
                      table_height=.74 - robot.entity_origin_pose[2])
        planner = PlannerProcess(config, Path(output)/f'planner_{robot.robot_name}.log')
        owned.append(planner)
        manager.planner[robot.robot_name] = manager.ik_solver[robot.robot_name] = planner
