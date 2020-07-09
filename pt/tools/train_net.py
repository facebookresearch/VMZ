from pathlib import Path
import uuid
import submitit
from vmz.common import utils
from vmz.func import opts


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from func import train

        self._setup_gpu_args()
        train.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        os.remove(self.args.dist_url[7:])  # remove file:// at the beginning
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        import os

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        if os.path.basename(self.args.output_dir) != str(job_env.job_id):
            self.args.output_dir = os.path.join(
                self.args.output_dir, str(job_env.job_id)
            )
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = opts.parse_args()

    # Note that the folder will depend on the
    # job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=utils.get_shared_folder(args.name) / "%j")
    num_gpus_per_node = 8
    executor.update_parameters(
        mem_gb=45 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=args.nodes,
        timeout_min=60 * 62,
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
    )

    executor.update_parameters(name=args.name)

    args.dist_url = utils.get_init_file(args.name).as_uri()
    args.output_dir = str(utils.get_shared_folder(args.name))
    trainer = Trainer(args)
    job = executor.submit(trainer)

    # job.task(0).result()


if __name__ == "__main__":
    main()
