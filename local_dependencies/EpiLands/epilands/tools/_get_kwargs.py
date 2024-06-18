# Getter Functions
def get_kwargs(items, **kwargs):
    items_in_kwargs = [item for item in items if item in kwargs]
    return {i: kwargs.get(i) for i in items_in_kwargs}
