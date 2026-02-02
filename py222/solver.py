#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import py222

hO = np.ones(729, dtype=np.int_) * 12
hP = np.ones(117649, dtype=np.int_) * 12
_tables_generated = False

moveStrs = {0: "U", 1: "U'", 2: "U2", 3: "R", 4: "R'", 5: "R2", 6: "F", 7: "F'", 8: "F2"}

# generate pruning table for the piece orientation states
def genOTable(s, d, lm=-3):
  index = py222.indexO(py222.getOP(s))
  if d < hO[index]:
    hO[index] = d
    for m in range(9):
      if int(m / 3) == int(lm / 3):
        continue
      genOTable(py222.doMove(s, m), d + 1, m)

# generate pruning table for the piece permutation states
def genPTable(s, d, lm=-3):
  index = py222.indexP(py222.getOP(s))
  if d < hP[index]:
    hP[index] = d
    for m in range(9):
      if int(m / 3) == int(lm / 3):
        continue
      genPTable(py222.doMove(s, m), d + 1, m)

# only generate the tables once
def _generate_tables():
  global _tables_generated
  if not _tables_generated:
    genOTable(py222.initState(), 0)
    genPTable(py222.initState(), 0)
    _tables_generated = True

# IDA* which prints all optimal solutions
def IDAStar(s, d, moves, lm=-3):
  if py222.isSolved(s):
    printMoves(moves)
    return True
  else:
    sOP = py222.getOP(s)
    if d > 0 and d >= hO[py222.indexO(sOP)] and d >= hP[py222.indexP(sOP)]:
      dOptimal = False
      for m in range(9):
        if int(m / 3) == int(lm / 3):
          continue
        newMoves = moves[:]; newMoves.append(m)
        solved = IDAStar(py222.doMove(s, m), d - 1, newMoves, m)
        if solved and not dOptimal:
          dOptimal = True
      if dOptimal:
        return True
  return False

# IDA* which returns the first optimal solution (as list of move indices)
def _IDAStarSilent(s, d, moves, lm=-3):
  if py222.isSolved(s):
    return moves[:]  # Return a copy of the solution
  else:
    sOP = py222.getOP(s)
    if d > 0 and d >= hO[py222.indexO(sOP)] and d >= hP[py222.indexP(sOP)]:
      for m in range(9):
        if int(m / 3) == int(lm / 3):
          continue
        newMoves = moves[:]; newMoves.append(m)
        result = _IDAStarSilent(py222.doMove(s, m), d - 1, newMoves, m)
        if result is not None:
          return result  # Return first solution found
  return None

# print a move sequence from an array of move indices
def printMoves(moves):
  moveStr = ""
  for m in moves:
    moveStr += moveStrs[m] + " "
  print(moveStr)

# solve a cube state and return the optimal solution as a list of move indices
def solve(s, verbose=False):
  """
  Solve a 2x2x2 cube state and return the optimal solution.
  
  Args:
    s: The cube state (24-element numpy array of sticker colors)
    verbose: If True, print progress information
    
  Returns:
    List of move indices representing the optimal solution.
    Move indices: 0=U, 1=U', 2=U2, 3=R, 4=R', 5=R2, 6=F, 7=F', 8=F2
    Returns empty list if already solved.
  """
  # FC-normalize stickers
  s = py222.normFC(s)
  
  # Ensure pruning tables are generated
  _generate_tables()
  
  # Check if already solved
  if py222.isSolved(s):
    return []
  
  # Run IDA*
  for depth in range(1, 12):
    if verbose:
      print("depth {}".format(depth))
    result = _IDAStarSilent(s, depth, [])
    if result is not None:
      return result
  
  return []  # Should never reach here for valid states

# solve a cube state (original verbose version)
def solveCube(s):
  # print cube state
  py222.printCube(s)

  # FC-normalize stickers
  print("normalizing stickers...")
  s = py222.normFC(s)

  # generate pruning tables
  print("generating pruning tables...")
  _generate_tables()

  # run IDA*
  print("searching...")
  solved = False
  depth = 1
  while depth <= 11 and not solved:
    print("depth {}".format(depth))
    solved = IDAStar(s, depth, [])
    depth += 1

if __name__ == "__main__":
  # input some scrambled state
  s = py222.doAlgStr(py222.initState(), "R U2 R2 F2 R' F2 R F R")
  # solve cube
  solveCube(s)

