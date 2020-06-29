from collections import defaultdict, deque
import datetime
import time

import torch
import torch.distributed as dist

from .utils import is_dist_avail_and_initialized, is_main_process


__all__ = [
    "SmoothedValue",
    "MetricLogger",
    "get_default_loggers",
    "get_default_loggers",
]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.ws = window_size

    def reset(self):
        self.__init__(window_size=self.ws, fmt=self.fmt)

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", writer=None, stat_set="train", epoch=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.metric_set = stat_set
        self.epoch = epoch

        self.writer = writer
        self.writer_step = 0

        self.meters["iter_time"] = SmoothedValue(fmt="{avg:.4f}")
        self.meters["data_time"] = SmoothedValue(fmt="{avg:.4f}")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def reset_meters(self):
        print("Logging: resseting all meters")
        for name, meter in self.meters.items():
            meter.reset()
        print(
            "Logging: resseting all meters done, updating epoch to {}".format(
                self.epoch + 1
            )
        )
        self.epoch += 1

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()

        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            self.meters["data_time"].update(time.time() - end)
            yield obj
            self.meters["iter_time"].update(time.time() - end)
            self._write_meters()
            if i % print_freq == 0:
                eta_seconds = self.meters["iter_time"].global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(self.meters["iter_time"]),
                            data=str(self.meters["data_time"]),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(self.meters["iter_time"]),
                            data=str(self.meters["data_time"]),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {}".format(header, total_time_str))
        self._write_epoch(total_time_str)

    def _write_meters(self):
        if self.writer is not None:
            for name, meter in self.meters.items():
                self.writer.add_scalar(
                    f"iter/{self.metric_set}_{name}", meter.avg, self.writer_step
                )

            self.writer_step += 1

    def _write_epoch(self, total_time_string):
        if self.writer is not None:
            for name, meter in self.meters.items():
                self.writer.add_scalar(
                    f"epoch/{self.metric_set}_{name}", meter.avg, self.epoch
                )

            self.writer.add_text(
                f"epoch/{self.metric_set}_totaltime", total_time_string, self.epoch
            )


def setup_tbx(save_dir):
    # from tensorboardX import SummaryWriter
    # this will be replaced by
    from torch.utils.tensorboard import SummaryWriter

    if not is_main_process():
        return None

    writer = SummaryWriter(save_dir)
    return writer


def get_default_loggers(writer, epoch):
    stat_loggers = dict()
    stat_loggers["train"] = MetricLogger(
        delimiter="  ", writer=writer, stat_set="train", epoch=epoch
    )
    stat_loggers["train"].add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    stat_loggers["train"].add_meter(
        "clips/s", SmoothedValue(window_size=10, fmt="{value:.3f}")
    )

    stat_loggers["val"] = MetricLogger(
        delimiter="  ", writer=writer, stat_set="val", epoch=epoch
    )

    return stat_loggers

