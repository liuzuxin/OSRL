from easy_runner import EasyRunner

if __name__ == "__main__":

    exp_name = "benchmark"
    runner = EasyRunner(log_name=exp_name)

    task = [
        # bullet safety gym envs
        "OfflineAntCircle-v0",
        "OfflineAntRun-v0",
        "OfflineCarCircle-v0",
        "OfflineDroneCircle-v0",
        "OfflineDroneRun-v0",
        "OfflineBallCircle-v0",
        "OfflineBallRun-v0",
        "OfflineCarRun-v0",
        # safety gymnasium: car
        "OfflineCarButton1Gymnasium-v0",
        "OfflineCarButton2Gymnasium-v0",
        "OfflineCarCircle1Gymnasium-v0",
        "OfflineCarCircle2Gymnasium-v0",
        "OfflineCarGoal1Gymnasium-v0",
        "OfflineCarGoal2Gymnasium-v0",
        "OfflineCarPush1Gymnasium-v0",
        "OfflineCarPush2Gymnasium-v0",
        # safety gymnasium: point
        "OfflinePointButton1Gymnasium-v0",
        "OfflinePointButton2Gymnasium-v0",
        "OfflinePointCircle1Gymnasium-v0",
        "OfflinePointCircle2Gymnasium-v0",
        "OfflinePointGoal1Gymnasium-v0",
        "OfflinePointGoal2Gymnasium-v0",
        "OfflinePointPush1Gymnasium-v0",
        "OfflinePointPush2Gymnasium-v0",
        # safety gymnasium: velocity
        "OfflineAntVelocityGymnasium-v1",
        "OfflineHalfCheetahVelocityGymnasium-v1",
        "OfflineHopperVelocityGymnasium-v1",
        "OfflineSwimmerVelocityGymnasium-v1",
        "OfflineWalker2dVelocityGymnasium-v1",
        # metadrive envs
        "OfflineMetadrive-easysparse-v0",
        "OfflineMetadrive-easymean-v0",
        "OfflineMetadrive-easydense-v0",
        "OfflineMetadrive-mediumsparse-v0",
        "OfflineMetadrive-mediummean-v0",
        "OfflineMetadrive-mediumdense-v0",
        "OfflineMetadrive-hardsparse-v0",
        "OfflineMetadrive-hardmean-v0",
        "OfflineMetadrive-harddense-v0",
    ]

    policy = ["train_bc", "train_bcql", "train_bearl", "train_coptidice", "train_cpq"]

    # Do not write & to the end of the command, it will be added automatically.
    template = "nohup python examples/train/{}.py --task {} --device cpu"

    train_instructions = runner.compose(template, [policy, task])
    runner.start(train_instructions, max_parallel=15)
