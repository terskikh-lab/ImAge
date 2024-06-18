from multiprocesstools import (
    MultiProcessHelper,
    RunTimeCounter,
    wait_until_file_exists,
    run_func_IO_loop,
)
from typing import Union, Callable
from functools import partial
import logging

logger = logging.getLogger(__name__)


class Module(MultiProcessHelper):
    def __init__(
        self,
        name,
        output_directory,
        loggers,
    ):
        super().__init__(
            name=name,
            working_directory=output_directory,
            loggers=loggers,
        )
        self._processes = {}

    @property
    def methods_list(self):
        return [
            method_name
            for method_name in dir(self)
            if not method_name.startswith("_") and callable(getattr(self, method_name))
        ]

    @property
    def processes_list(self):
        return [process for process in self._processes.keys()]

    def add_process(self, function: Callable, *args, **kwargs):
        if isinstance(function, Callable):
            partial_func = partial(function, *args, **kwargs)
            partial_func.__name__ = function.__name__
            if hasattr(function, "input_tag") and hasattr(function, "output_tag"):
                partial_func.input_tag = function.input_tag
                partial_func.output_tag = function.output_tag
            else:
                raise AttributeError(
                    f"input_tag or output_tag not specified for given process"
                )
            self._processes[function.__name__] = partial_func
        else:
            raise ValueError(
                f"function must be a Callable but {type(function)} was given"
            )

    def get_process(
        self,
        process,
    ):
        if process in self._processes.keys():
            return self._processes[process]
        else:
            print(
                f"process not in processes.\n\nGiven:{process} Processes:\n\n{self.processes_list}"
            )

    def remove_process(self, process):
        if process in self._processes.keys():
            del self._processes[process]
        else:
            print(
                f"process not in processes.\n\nGiven:{process} Processes:\n\n{self.processes_list}"
            )

    def clear_all_processes(self):
        self._processes = {}

    def run_all_processes(self):
        for name, process in self._processes.items():
            logger.info(f"Running process {name}")
            kwargs = {}
            for attr in process.input_tag:
                if attr is None:
                    continue
                if hasattr(self, attr):
                    kwargs[attr] = self.__getattribute__(attr)
                else:
                    raise AttributeError(
                        f"Invalid input tag: {self.input_tag}. Attr {attr} does not exist"
                    )
            output = process(**kwargs)
            if (len(process.output_tag) != 1) and (
                len(output) != len(process.output_tag)
            ):
                raise ValueError(
                    f"Output length does not match tag: output_tag: {process.output_tag} ({len(process.output_tag)}) but the process returned {len(output)} args"
                )
            for i, attr in enumerate(process.output_tag):
                outi = output[i] if len(process.output_tag) > 1 else output
                if attr is None:
                    continue
                if hasattr(self, attr):
                    logger.warning(f"Updating {attr}...")
                    self.__setattr__(attr, outi)
                else:
                    self.__setattr__(attr, outi)
                # else:
                #     raise AttributeError(f"Invalid output tag: {process.output_tag}. Attr {attr} does not exist")
