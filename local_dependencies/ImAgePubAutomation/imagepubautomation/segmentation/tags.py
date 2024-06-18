from multiprocesspipelines import tag_factory

outputs_image_data = tag_factory("outputs_image_data", "output", "image_data")
outputs_segmentation_image = tag_factory("outputs_segmentation_image", "output", "segmentation_image")
outputs_masks_objects_details = tag_factory("outputs_masks_objects_details", "output", "masks_objects_details")

receives_image_data = tag_factory("receives_image_data", "input", "image_data")
receives_segmentation_image = tag_factory("receives_segmentation_image", "input", "segmentation_image")
receives_masks_objects_details = tag_factory("receives_masks_objects_details", "input", "masks_objects_details")
