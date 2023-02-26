IS_POSITIVE = lambda x: x > 0
IS_BETWEEN_0_AND_1 = lambda x: x > 0 and x <= 1

def validate(value, value_type, is_valid=lambda *_: True, default='default'):
    assert (type(value) is value_type), f'Type should be {value_type} but it is {type(value)}'

    if is_valid(value):
        return value
    else:
        assert (default != 'default')
        return default