"""Checks reject misparsed decisions and unsupported citation syntax before inference."""
from run import explanation, parse_label

assert parse_label('Output:\n0') == 0
assert parse_label('Output:\n1') == 1
assert parse_label('Output:\n2') is None
assert parse_label('Output: 1\nExtra words') is None
assert parse_label('Output: 0\nOutput: 1') is None
good = 'Explanation: {"observations":[{"claim":"A car moves.","frames":[1,8]}],"unresolved":["Intent"]}\nOutput: 1'
assert explanation(good) is not None
assert explanation(good.replace('[1,8]', '[0,9]')) is None
assert explanation(good.replace('[1,8]', '[true]')) is None
assert explanation(good.replace('[1,8]', '[1,1]')) is None
assert explanation(good.replace('"Intent"', 'null')) is None
assert explanation(good.replace('"observations"', '"unsupported_key"')) is None
print('11 binary/schema contract checks passed')
