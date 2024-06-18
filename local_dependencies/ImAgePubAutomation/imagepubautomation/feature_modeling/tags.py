from multiprocesspipelines import tag_factory

outputs_model = tag_factory("outputs_model", "output", "model")
outputs_data = tag_factory("outputs_data", "output", "data")
outputs_distance_matrix = tag_factory(
    "outputs_distance_matrix", "output", "outputs_distance_matrix"
)
outputs_platemap = tag_factory("outputs_platemap", "output", "platemap")

receives_output_directory = tag_factory(
    "receives_output_directory", "input", "output_directory"
)
receives_model = tag_factory("receives_model", "input", "model")
receives_data = tag_factory("receives_data", "input", "data")
receives_distance_matrix = tag_factory(
    "receives_distance_matrix", "input", "distance_matrix"
)
receives_platemap = tag_factory("receives_platemap", "input", "platemap")


# outputs_model = tag_factory("outputs_model", "output", "model")
# outputs_features = tag_factory("outputs_features", "output", "features")
# outputs_observations = tag_factory("outputs_observations", "output", "observations")
# outputs_platemap = tag_factory("outputs_platemap", "output", "platemap")

# receives_output_directory = tag_factory("receives_output_directory", "input", "output_directory")
# receives_model = tag_factory("receives_model", "input", "model")
# receives_features = tag_factory("receives_features", "input", "features")
# receives_platemap = tag_factory("receives_platemap", "input", "platemap")
# receives_observations = tag_factory("receives_observations", "input", "observations")
