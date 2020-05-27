from gym.envs.registration import register

register(
    id='surface_seg-v0',
    entry_point='surface_seg.envs:MCSEnv',
)
