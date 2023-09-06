

def map_range(value, old_min, old_max, new_min, new_max):
    old_range = old_max - old_min
    new_range = new_max - new_min
    new_value = (((value - old_min) * new_range) / old_range) + new_min
    return new_value

value = -9
old_min = -10
old_max = 10
new_min = 10
new_max = 20

mapped_value = map_range(value, old_min, old_max, new_min, new_max)
print(mapped_value)
