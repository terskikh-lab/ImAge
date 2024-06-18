from multiprocesspipelines import tag_factory, variable_attribute_tag_factory

outputs_nothing = tag_factory("outputs_nothing", "output", None)
receives_nothing = tag_factory("receives_nothing", "input", None)

recieves_variable_attribute = variable_attribute_tag_factory("variable_input", "input")
outputs_variable_attribute = variable_attribute_tag_factory("variable_output", "output")