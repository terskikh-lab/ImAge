from multiprocesspipelines import tag_factory

outputs_objects = tag_factory("outputs_objects", "output", "objects")
outputs_details = tag_factory("outputs_details", "output", "details")
outputs_features = tag_factory("outputs_features", "output", "features")

receives_objects = tag_factory("receives_objects", "input", "objects")
receives_details = tag_factory("receives_details", "input", "details")
receives_features = tag_factory("receives_features", "input", "features")


