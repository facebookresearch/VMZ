### This is submission script
import submitit
from vmz.common import utils
from vmz.func import opts

# TODO: docs


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from vmz.func.feats import ef_main

        self._setup_gpu_args()
        ef_main(self.args)

    def checkpoint(self):
        import submitit

        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        import os

        job_env = submitit.JobEnvironment()
        if os.path.basename(self.args.output_dir) != str(job_env.job_id):
            self.args.output_dir = os.path.join(
                self.args.output_dir, str(job_env.job_id)
            )


def main():
    args = opts.parse_args()
    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=utils.get_shared_folder(args.name) / "%j")
    num_gpus_per_node = 8
    args.batch_size = args.batch_size * num_gpus_per_node
    executor.update_parameters(
        mem_gb=45 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        # tasks_per_node=1,  # one task per GPU
        cpus_per_task=80,
        nodes=1,
        timeout_min=60 * 16,
        # Below are cluster dependent parameters
        slurm_partition="dev",
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
