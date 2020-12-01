"""
picross.py
RJP - 2020-12-12

A Picross/Nonogram solver
"""

import copy
from enum import IntEnum
import sys


LOG_TAGS = ['INFO']
VALUE_CHARS = [' ', 'X', '\u2588']

def LOG(tag, *args, **kwargs):
  """Debug logging. takes a tag and only prints if the tag is in the global LOG_TAGS"""
  if tag in LOG_TAGS:
    print(*args, **kwargs)


class Value(IntEnum):
  UNSOLVED = 0
  X = 1
  FILL = 2

class UnitType(IntEnum):
  ROW = 0
  COL = 1

  
class UnitInfo:
  def __init__(self, unit_pos, unit_type, unit_length):
    self.pos = unit_pos
    self.type = unit_type
    self.length = unit_length


class Group:
  """Group represents the solution to a single hint in a unit (row or column)"""
  def __init__(self, length, position_set, group_id, info, known_values=set(), known_xs=set(), solution='*'):
    """Constructor will validate the given state, synthesize some useful
    properties, and apply some basic constraints (like known values when the
    range is < 2*length)"""
    self.length = length
    self.id = group_id
    self.info = info
    self.solution = solution
    span_sets = [set(range(position, position + length)) for position in position_set]
    self.valid_span_sets = list(filter(lambda span_set: (all(x not in span_set for x in known_xs)), span_sets))
    self.valid_range = set.union(*self.valid_span_sets) if self.valid_span_sets else set()
    self.position_set = {min(span_set) for span_set in self.valid_span_sets}
    self.uncertainty = len(self.position_set) - 1
    self.is_solved = self.uncertainty == 0
    self.known_values = (known_values | (set.intersection(*self.valid_span_sets) if len(self.valid_span_sets) > 0 else set())) & self.valid_range
    self.known_xs = known_xs | ({min(self.position_set) - 1, min(self.position_set) + length} - {-1, info.length} if self.is_solved else set())
    self.has_contradiction = len(self.position_set) == 0
    self.updated_positions = (self.known_values - known_values) | (self.known_xs - known_xs)
    # synthesized value to help with searching
    self.weight = length - self.uncertainty + len(self.known_values)

  def constrain_range(self, new_range, known_values, known_xs):
    """A helper function called by higher-order classess to construct a
    derivative of the group from higher-order constraints"""
    valid_span_sets = list(filter(lambda span_set: (span_set & new_range == span_set), self.valid_span_sets))
    position_set = {min(span_set) for span_set in valid_span_sets}
    return Group(self.length, position_set, self.id, self.info, known_values, known_xs)

  def is_same(self, other):
    """A comparison helper for higher-order classes to determine if applying
    a contraint lead to new information about the puzzle"""
    if self.position_set != other.position_set or self.known_values != other.known_values or self.known_xs != other.known_xs:
      return False
    return True

  def __repr__(self):
    return f'{"!" if self.has_contradiction else ""}{"R" if self.info.type == UnitType.ROW else "C"}{self.info.pos}.{self.id}[{self.length}]{self.position_set}'


class Unit:
  """Unit represents a row or column and contains the Groups representing
  each hint in the unit"""
  def __init__(self, info, groups, known_values=set(), known_xs=set(), solution='*'):
    self.info = info
    self.groups = groups
    self.solution = solution
    self.uncertainty = sum(group.uncertainty for group in groups)
    self.is_solved = all(group.is_solved for group in groups) or len(groups) == 0

    # mutable state - may be modified in Unit.apply_constraints() and will be
    # used in Puzzle.update() to construct a new puzzle state
    self.known_values = known_values.union(*[group.known_values for group in groups])
    self.known_xs = known_xs.union(*[group.known_xs for group in groups])
    self.has_contradiction = any(group.has_contradiction for group in groups)
    self.contradiction = ''
    self.updated_groups = copy.deepcopy(self.groups)
    # Used to track whether we made progress or not
    self.has_updated_groups = False

    # Heavy operation, but has the meat of the logic/optimizations
    self.apply_constraints()

    # Used to track whether we made progress or not
    self.updated_positions = (self.known_values - known_values) | (self.known_xs - known_xs)

  def apply_constraints(self):
    """Apply human-centric strategies to infer puzzle solution without any guess-and-test"""
    # simple constraints mostly determine if we have a contradiction and let us
    # early-out from wasting time on more expensive constratints. Some may
    # modify known_values, known_xs
    contradiction_msg = f'{"!" if self.has_contradiction else ""}U{"R" if self.info.type == UnitType.ROW else "C"}{self.info.pos}'
    simple_constraints = [
      self.UC_IsSolved,
      self.UC_ValueConflict,
      self.UC_TooManyValues,
      self.UC_TooManyXs,
    ]
    for constraint in simple_constraints:
      msg = constraint() or ''
      if self.has_contradiction:
        self.contradiction =  f'{contradiction_msg}- {msg}'
        return
      if self.is_solved:
        return

    # constraints that require higher-order data and may produce updated groups 
    expensive_constraints = [
      self.UC_SpanLengths,
      self.UC_BookendRange,
      self.UC_AssignSpans,
      self.UC_FillXs,
      self.UC_Partitions,
    ]
    data = Unit.ConstraintData(self)
    for constraint in expensive_constraints:
      msg = constraint(data) or ''
      if self.has_contradiction:
        self.contradiction =  f'{contradiction_msg}- {msg}'
        break
      if self.is_solved:
        break

  def UC_IsSolved(self):
    """All groups are solved. Pad out the rest of the unit with X's"""
    if self.is_solved:
      self.known_xs = set(range(self.info.length)) - self.known_values
    if self.has_contradiction:
      return ','.join(str(group) for group in self.groups if group.has_contradiction)
  
  def UC_ValueConflict(self):
    """Make sure there is no disagreement between which squares are X's and which are filled"""
    if len(self.known_xs & self.known_values) > 0:
      self.has_contradiction = True
      return f'!badXsOs(x{self.known_xs}o{self.known_values})'

  def UC_TooManyValues(self):
    """Make sure we don't have too many filled squares for the hints we have"""
    if len(self.known_values) > sum(group.length for group in self.groups):
      self.has_contradiction = True
      return f'!tooManyValues({self.known_values} > {"+".join(str(group.length) for group in self.groups)})'

  def UC_TooManyXs(self):
    """Make sure we don't have too many X's to actually solve our hints"""
    if self.info.length - len(self.known_xs) < sum(group.length for group in self.groups):
      self.has_contradiction = True
      return f'!tooManyValues({self.info.length} - {self.known_xs} < {"+".join(str(group.length) for group in self.groups)})'

  class ConstraintData:
    def __init__(self, unit):
      """ConstraintData represents some synthesized values that will help apply higher-order strategies"""
      # spans are contiguous filled squares, we want to try to assign them to a
      # group to narrow down where everything should fit
      self.spans = []
      span_start = None
      for i in range(unit.info.length):
        if span_start is not None and i not in unit.known_values:
          self.spans.append([span_start, i-1])
          span_start = None
        if span_start is None and i in unit.known_values:
          span_start = i
      if span_start is not None:
        self.spans.append([span_start, unit.info.length-1])
      self.span_sets = [set(range(span[0], span[1]+1)) for span in self.spans]
      self.span_lengths = [span[1] - span[0] + 1 for span in self.spans]

      # partitions are the spaces between X's. We can use them to narrow in on
      # where to place a group, rule out other groups that can't be fit into a
      # partition, or prove a contradiction
      self.partitions = []
      non_x_set = set(range(unit.info.length)) - unit.known_xs
      partition_start = None
      for i in range(unit.info.length):
        if partition_start is not None and i not in non_x_set:
          self.partitions.append([partition_start, i-1])
          partition_start = None
        if partition_start is None and i in non_x_set:
          partition_start = i
      if partition_start is not None:
        self.partitions.append([partition_start, unit.info.length-1])
      self.partition_sets = [set(range(partition[0], partition[1]+1)) for partition in self.partitions]
      self.partition_lengths = [partition[1] - partition[0] + 1 for partition in self.partitions]

      # tracks how many groups could potentially own a span
      self.matching_groups_list = []
      for i, span_set in enumerate(self.span_sets):
        self.matching_groups_list.append([group for group in unit.updated_groups if self.span_lengths[i] <= group.length and span_set & group.valid_range == span_set])

  def UC_SpanLengths(self, data):
    """Make sure we don't have a span that belongs to no group -
    typically one longer than any group in the unit"""
    for i, matching_groups in enumerate(data.matching_groups_list):
      if len(matching_groups) == 0:
        self.has_contradiction = True
        return f'!spanLengths({data.span_sets[i]})'

  def UC_BookendRange(self, data):
    """When we have knowledge about where a group should start, make sure
    it's neighbors are shifted accordingly. i.e. If we prove the first group
    begins 3 squares from the begging of a row, we can probably push the
    range of every other group in the unit at least 3 squares to the right.
    This is important to know when we can assign a middle-ish span to a
    specific group. It also lets us pad X's before and after a bunch of
    groups."""

    new_range = set(range(self.info.length))
    # left to right
    for i, group in enumerate(self.updated_groups):
      updated_group = group.constrain_range(new_range, self.known_values, self.known_xs)
      if updated_group.has_contradiction:
        self.has_contradiction = True
        return f'!badBookends({updated_group})'
      if not updated_group.is_same(group):
        self.has_updated_groups
        self.updated_groups[i] = updated_group
      new_range &= set(range(max(min(updated_group.valid_range), min(new_range)) + group.length + 1, self.info.length))

    # right to left
    new_range = set(range(self.info.length))
    max_group = len(self.updated_groups) - 1
    for i, group in enumerate(reversed(self.updated_groups)):
      updated_group = group.constrain_range(new_range, self.known_values, self.known_xs)
      if updated_group.has_contradiction:
        self.has_contradiction = True
        return f'!badBookends({updated_group})'
      if not updated_group.is_same(group):
        self.has_updated_groups
        self.updated_groups[max_group - i] = updated_group
      new_range &= set(range(0, min(max(updated_group.valid_range), max(new_range)) - group.length))

  def UC_AssignSpans(self, data):
    """Try to prove which hints some filled squares are part of. This cuts
    out big swathes of uncertainty in where to position each group in a unit"""
    # left to right and right to left, if can pin a span to a set, we can constrain the range of other groups
    for i, matching_groups in enumerate(data.matching_groups_list):
      if len(matching_groups) == 1:
        match_group = matching_groups[0]
        span = data.span_sets[i]
        length = match_group.length
        new_range = set(range(min(span), min(span) + length)) | set(range(max(span) - length + 1, max(span))) & set(range(self.info.length))
        if new_range != match_group.valid_range:
          updated_group = match_group.constrain_range(new_range, self.known_values, self.known_xs)
          if updated_group.has_contradiction:
            self.has_contradiction = True
            return f'!badSpans(!{updated_group})'

          left_range = set(range(0, max(updated_group.valid_range) - updated_group.length))
          right_range = set(range(min(updated_group.valid_range) + updated_group.length + 1, self.info.length))
          left_neighbors = [group.constrain_range(left_range, self.known_values, self.known_xs) for group in self.updated_groups[:match_group.id]]
          right_neighbors = [group.constrain_range(right_range, self.known_values, self.known_xs) for group in self.updated_groups[match_group.id+1:]]
          updated_groups = left_neighbors + [updated_group] + right_neighbors

          for j, group in enumerate(updated_groups):
            if group.has_contradiction:
              self.has_contradiction = True
              return f'!badSpans(!{group})'
            if not group.is_same(self.updated_groups[j]):
              self.has_updated_groups = True
              self.updated_groups[j] = group

    if self.has_updated_groups:
      self.known_values = self.known_values.union(*[group.known_values for group in self.updated_groups])
      self.known_xs = self.known_xs.union(*[group.known_xs for group in self.updated_groups])

  def UC_FillXs(self, data):
    """Plug gaps between possible group placements with X's. Helps a lot with partitioning."""
    unit_values_set = set.union(*(group.valid_range for group in self.updated_groups) if self.updated_groups else set())
    fill_xs = set(range(self.info.length)) - unit_values_set
    self.known_xs |= fill_xs

  def UC_Partitions(self, data):
    """Try to fit groups in the spaces between X's"""
    groups_in_partitions = []
    for i, partition_set in enumerate(data.partition_sets):
      groups_in_partitions.append([group for group in self.updated_groups if len(group.valid_range & partition_set) >= group.length])
    partitions_in_groups = [[]]*len(self.updated_groups)
    for i, groups in enumerate(groups_in_partitions):
      for group in groups:
        partitions_in_groups[group.id].append(i)
    for i, groups in enumerate(groups_in_partitions):
      if len(groups) == 1 and len(partitions_in_groups[groups[0].id]) == 1:
        partition = data.partition_sets[i]
        group = groups[0]
        updated_group = group.constrain_range(partition, self.known_values, self.known_xs)
        if not group.is_same(updated_group):
          self.has_updated_groups = True
          self.updated_groups[group.id] = updated_group
    if self.has_updated_groups:
      self.known_values = self.known_values.union(*[group.known_values for group in self.updated_groups])
      self.known_xs = self.known_xs.union(*[group.known_xs for group in self.updated_groups])

  def __repr__(self):
    return f'{"!" if self.has_contradiction else ""}U{"R" if self.info.type == UnitType.ROW else "C"}{self.info.pos}[{self.groups}]'

class Puzzle:
  def __init__(self, group_vals_lists, units_list):
    self.group_vals_lists = group_vals_lists
    self.units_list = units_list
    self.solution = '*'
    self.all_units = units_list[UnitType.ROW] + units_list[UnitType.COL]
    self.all_groups = [group for unit in self.all_units for group in unit.groups]
    self.has_contradiction = any(unit.has_contradiction for unit in self.all_units)
    self.contradiction = ','.join(unit.contradiction for unit in self.all_units if unit.has_contradiction)
    self.is_solved = not self.has_contradiction and all(unit.is_solved for unit in self.all_units)
    self.uncertainty = sum(unit.uncertainty for unit in self.all_units)
    self.updated_positions_set = set()
    for unit in self.all_units:
      if len(unit.updated_positions) > 0:
        self.updated_positions_set |= {Puzzle.pos_coord(unit.info, position) for position in unit.updated_positions}

  def __repr__(self):
    return f'{"!" if self.has_contradiction else ""}P{self.uncertainty} - {self.solution}{self.contradiction}'

  class AssignInfo:
    def __init__(self, group, position):
      self.group = group
      self.position = position

    def __repr__(self):
      return f'{"R" if self.group.info.type == UnitType.ROW else "C"}{self.group.info.pos}.{self.group.id}@{self.position}'

  def update(self, assign_info=None, solution='*'):
    """Optionally assign a group position for guess-and-test DFS and
    recursively propagate constraints through the puzzle. Propagation stops
    when a contradiction is found or no progress is made solving the puzzle"""
    if len(self.updated_positions_set) == 0 and not assign_info:
      return self
    units_list = [[], []]
    for unit_type in [UnitType.ROW, UnitType.COL]:
      for unit in self.units_list[unit_type]:
        known_values = self.get_known_values(unit.info)
        known_xs = self.get_known_xs(unit.info)
        groups = [Group(group.length, group.position_set, group.id, group.info, known_values, known_xs, solution) for group in unit.updated_groups]
        if assign_info and assign_info.group.info.type == unit_type and unit.info.pos == assign_info.group.info.pos:
          group = groups[assign_info.group.id]
          new_group = Group(group.length, {assign_info.position}, group.id, group.info, known_values, known_xs, solution)
          groups[assign_info.group.id] = new_group

        units_list[unit_type].append(Unit(unit.info, groups, known_values, known_xs, solution))

    new_puzzle = Puzzle(self.group_vals_lists, units_list)
    new_puzzle.solution = solution
    if new_puzzle.has_contradiction:
      return new_puzzle
    return new_puzzle.update()

  def get_known_values(self, unit_info):
    """Get squares known to be filled. Takes them from the current unit and
    each intersecting unit. The intersecting units may have new information
    which propagating constraints from an update"""
    other_type = UnitType.COL if unit_info.type == UnitType.ROW else UnitType.ROW
    intersecting_values = [{i} if self.units_list[other_type][i].known_values & {unit_info.pos} else {} for i in range(unit_info.length)]
    return self.units_list[unit_info.type][unit_info.pos].known_values.union(*intersecting_values)

  def get_known_xs(self, unit_info):
    """Get squares known to be X's. Takes them from the current unit and
    each intersecting unit. The intersecting units may have new information
    which propagating constraints from an update"""
    other_type = UnitType.COL if unit_info.type == UnitType.ROW else UnitType.ROW
    intersecting_xs = [{i} if self.units_list[other_type][i].known_xs & {unit_info.pos} else {} for i in range(unit_info.length)]
    return self.units_list[unit_info.type][unit_info.pos].known_xs.union(*intersecting_xs)

  @staticmethod
  def Parse(path):
    """Reads in a text file col(top to bottom one per line) [empty line] (row
    left to right one per line) no trailing spaces"""
    with open(path) as f:
      lines = f.readlines()
    cols = []
    rows = []
    cols_done = False
    for line in lines:
      if not cols_done:
        if len(line.strip()) > 0:
          cols.append([int(n) for n in line.strip().split(" ")])
        else:
          cols_done = True
      else:
        if len(line.strip()) > 0:
          rows.append([int(n) for n in line.strip().split(" ")])
    group_vals_lists = [rows, cols]
    units_list = Puzzle.make_units_list(group_vals_lists)
    puzzle = Puzzle(group_vals_lists, units_list).update()
    return puzzle.update()

  @staticmethod
  def make_units_list(group_vals_lists):
    """creates the initial state of the puzzle - units_list[2] - [row,col],
    each unit populated with groups whose range is pre-constrained by their
    neighbors"""
    units_list = [[], []]
    assert len(group_vals_lists) == 2
    unit_lengths = [len(group_vals_lists[1]), len(group_vals_lists[0])]
    for unit_type, group_vals_list in enumerate(group_vals_lists):
      for unit_pos, group_vals in enumerate(group_vals_list):
        info = UnitInfo(unit_pos, UnitType(unit_type), unit_lengths[unit_type])
        unit_position_info = Puzzle.unit_from_group_vals(group_vals, info)
        units_list[unit_type].append(unit_position_info)
    return units_list

  @staticmethod
  def unit_from_group_vals(group_vals, info):
    """creates an initialized unit with groups pre-constrained by their neighbors"""
    groups = []
    offset = 0
    for i, val in enumerate(group_vals):
      if val == 0:
        continue
      rest = group_vals[i+1:]
      offset_max = info.length - sum(rest) - val - len(rest)
      position_set = set(range(offset, offset_max + 1))
      assert len(position_set) >= 1
      groups.append(Group(val, position_set, i, info))
      offset += val + 1
    return Unit(info, groups)

  @staticmethod
  def solve(puzzle):
    return Puzzle.search(puzzle)

  @staticmethod
  def search(puzzle):
      """Using depth-first search and propagation, try all possible values."""
      if not puzzle or puzzle.has_contradiction:
          LOG('SEARCH', f'{puzzle}')
          return False ## Failed earlier
      if puzzle.is_solved: 
          return puzzle ## Solved!
      LOG('SEARCH', f'{puzzle.solution}')
      puzzle.display('SEARCH')
      ## Chose the unfilled square s with the fewest possibilities
      assign_infos = puzzle.find_good_group_assignment()
      LOG('SEARCH', f'Trying {puzzle.solution}->{assign_infos}')
      for assign_info in assign_infos:
        solution = f'{puzzle.solution}->{assign_info}'
        new_puzzle = puzzle.update(assign_info, solution)
        new_puzzle.solution = solution
        if new_puzzle.has_contradiction:
          LOG('SEARCH', f'Dead End {solution} - {new_puzzle.contradiction}')
          new_puzzle.display('SEARCH')
          continue
        solved_puzzle = puzzle.search(new_puzzle)
        if solved_puzzle and solved_puzzle.is_solved:
          return solved_puzzle
      return False

  def find_good_group_assignment(self):
    """Find the most impactful group to guess and test for depth-first
    search. Hueristic is a synthetic value, weight which considers length of
    a unit and the number possible positions it could be in"""
    max_unit_length = max(len(self.units_list[UnitType.ROW]), len(self.units_list[UnitType.COL]))
    max_weight_value = -max_unit_length
    max_weight_group = None
    for group in self.all_groups:
      if not group.is_solved:
        if group.weight > max_weight_value:
          max_weight_value = group.weight
          max_weight_group = group
    return [Puzzle.AssignInfo(max_weight_group, position) for position in max_weight_group.position_set]

  def display(self, log_tag = 'INFO'):
    unit_values_list = self.get_unit_values_list()
    max_col_items = max([len(n) for n in self.group_vals_lists[UnitType.COL]])
    max_row = max([len(' '.join([str(j) for j in n])) for n in self.group_vals_lists[UnitType.ROW]])
    col_head_lines = [''] * max_col_items
    for col in self.group_vals_lists[UnitType.COL]:
      just = max_col_items - len(col)
      for i in range(just):
        col_head_lines[i] += '{:2s}'.format('')
      for i, item in enumerate(col):
        col_head_lines[i + just] += '{:2d}'.format(item)
    row_head_lines = [(' '.join([str(j) for j in n])).rjust(max_row) for n in self.group_vals_lists[UnitType.ROW]]
    max_row = max_row

    for col_head in col_head_lines:
      LOG(log_tag, ' ' * max_row + col_head)
    for i, row_head in enumerate(row_head_lines):
      if i % 5 == 0:
        LOG(log_tag, ' '* max_row + '+' + '-+' * len(self.group_vals_lists[UnitType.COL]))
      print_values = [VALUE_CHARS[v] for v in unit_values_list[UnitType.ROW][i]]
      LOG(log_tag, row_head + '┃', end='')
      for j in range(0, len(unit_values_list[UnitType.COL]), 5):
        LOG(log_tag, '|'.join([v for v in print_values[j:j+5]]), end='')
        LOG(log_tag, '┃', end='')
      LOG(log_tag, '')
    LOG(log_tag, ' '* max_row + '+' + '-+' * len(self.group_vals_lists[UnitType.COL]))

  def get_unit_values_list(self):
    unit_values_list = [[], []]
    for unit_type, units in enumerate(self.units_list):
      for unit in units:
        values = [Value.UNSOLVED] * unit.info.length
        for x in unit.known_xs:
          values[x] = Value.X
        for v in unit.known_values:
          values[v] = Value.FILL
        unit_values_list[unit_type].append(values)
    for unit_type, unit_values in enumerate(unit_values_list):
      other_type = UnitType.COL if UnitType(unit_type) == UnitType.ROW else UnitType.ROW
      for unit_pos, values in enumerate(unit_values):
        for i, val in enumerate(values):
          if val != Value.UNSOLVED and unit_values_list[other_type][i][unit_pos] == Value.UNSOLVED:
            unit_values_list[other_type][i][unit_pos] = val
    return unit_values_list

  @staticmethod
  def pos_coord(unit_info, position):
    return (unit_info.pos if unit_info.type == UnitType.ROW else position, unit_info.pos if unit_info.type == UnitType.COL else position)


def main():
  puzzle_path = sys.argv[1]
  print(f'Puzzle {puzzle_path}')
  solution = Puzzle.solve(Puzzle.Parse(puzzle_path))
  if solution:
    solution.display()
  else:
    print('Solution not found')

if __name__ == "__main__":
  main()
