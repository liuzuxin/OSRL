from fsrl.utils.exp_util import ExperimentGrid

if __name__ == "__main__":

    exp_name = "benchmark"
    runner = ExperimentGrid(log_name=exp_name)

    task = [
        "offline-AntCircle-v0", "offline-AntRun-v0", "offline-CarCircle-v0",
        "offline-DroneCircle-v0", "offline-DroneRun-v0"
    ]
    # outliers_percent = [0.05, 0.1, 0.15]
    # noise_scale = [0.05, 0.1, 0.15]
    policy = ["train_bc", "train_bcql", "train_bearl", "train_coptidice", "train_cpq"]
    # seed = [0, 10, 20]

    # Do not write & to the end of the command, it will be added automatically.
    template = "nohup python examples/train/{}.py --task {} --device cpu"

    train_instructions = runner.compose(template, [policy, task])
    runner.run(train_instructions, max_parallel=15)
