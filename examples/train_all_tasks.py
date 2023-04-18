from saferl.utils.exp_util import ExperimentGrid

if __name__ == "__main__":

    exp_name = "benchmark"
    runner = ExperimentGrid(log_name=exp_name)

    task = ["offline-CarCircle-v0"]
    policy = ["train_bc", "train_bcql", "train_bearl", "train_coptidice", "train_cpq"]
    seed = [0, 10, 20]

    # Do not write & to the end of the command, it will be added automatically.
    template = "nohup python examples/train/{}.py --task {} --seed {}"

    train_instructions = runner.compose(template,
                                        [policy, task, seed])
    runner.run(train_instructions, max_parallel=15)
