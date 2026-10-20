"""Enable full-precision aligned depth before native camera creation."""
CAMERA_ALIASES = dict(cam_high=('cam_head', 'cam_high', 'head_camera', 'top_camera'),
                      cam_left_wrist=('cam_left_wrist', 'left_camera'),
                      cam_right_wrist=('cam_right_wrist', 'right_camera'))


def enable_camera_depth(cfg):
    from omegaconf import OmegaConf, open_dict
    with open_dict(cfg):
        vision = cfg.eval_cfg.observation.vision
        vision.depth = True
        vision.approximate_depth = False
        for camera, aliases in CAMERA_ALIASES.items():
            names = [key for key in aliases if key in cfg.camera]
            if not names and camera in ('cam_left_wrist', 'cam_right_wrist'):
                # Native RobotManager._load_camera_cfg adds wrist geometry
                # inside create_eval_env, after this function runs. Configure
                # its canonical annotator now, without creating a dummy mount.
                # The session checks that both actual RGB-D sensors exist.
                names = [camera]
            if len(names) != 1:
                raise ValueError(f'Expected exactly one configured {camera} camera')
            name = names[0]
            common = cfg.camera.annotator.get(name, cfg.camera.annotator.get('common'))
            if common is None:
                raise ValueError(f'Missing {camera} annotator configuration')
            capture = OmegaConf.to_container(common, resolve=True)
            capture.update(enabled=True, distance_to_image_plane_capture=dict(type='distance_to_image_plane', device='cpu'))
            cfg.camera.annotator[name] = capture
