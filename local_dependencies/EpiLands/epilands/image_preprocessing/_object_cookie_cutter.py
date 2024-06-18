import numpy as np
from typing import List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from ._extract_object import extract_object


def object_cookie_cutter(
    image: np.ndarray,
    mask: np.ndarray,
    objects: List[Union[int, float]],
) -> Tuple[dict, dict]:
    # start=time.perf_counter() # Use these for testing the speed of the function
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            extract_object,
            repeat(image),
            repeat(mask),
            objects,
        )
    object_images = {}
    object_masks = {}
    for object, result in zip(objects, results):
        object_images[object] = result[0]
        object_masks[object] = result[1]
    # finish = time.perf_counter()
    # print('Finished in {} seconds'.format(finish-start))
    # return finish-start
    return object_images, object_masks
