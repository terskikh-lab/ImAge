import tensorflow.config as tfc
import random
import platform


def gpuinit(gpuN=None):
    if isinstance(gpuN, str):
        gpuN = int(gpuN)
    if platform.system() != "Darwin":
        gpus = tfc.list_physical_devices(device_type="GPU")
        if gpuN==-1:
            tfc.set_visible_devices([], 'GPU')
        else:
            if gpuN != None:
                if gpuN > len(gpus):
                    raise ValueError(
                        f"Cannot assign GPU number {gpuN} because there aren't enough GPUs. List of GPUs available:\n\n{gpus}"
                    )
                gpuIdx = gpuN
            else:
                gpuIdx = random.sample(list(range(0, len(gpus))), 1)[0]
            tfc.experimental.set_memory_growth(gpus[gpuIdx], True)
            tfc.set_visible_devices(gpus[gpuIdx], "GPU")
