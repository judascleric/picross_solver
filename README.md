# Picross Solver

Written after playing a bunch of Picross S4 on Nintendo switch, probably my
6th picross game. It was getting pretty mechanical and I wanted to explore
how to programmatically solve the puzzles. I originally wanted to see if I
could apply Peter Norvig's Sudoku solution with depth-first search and
constraint propagation since it was so short and elegant
(https://norvig.com/sudoku.html). I couldn't quite make the leap on a cold
start so I started programming constraints in the human way that I solved it
rather than in a compute-centric solution. When I finally had my head around
the problem space and the data, I made a simple and elegant v2... which ran
like a dog. It could only solve small puzzles in a reasonable amount of time.
I probably could have switched to Numpy for speed or reworked the data to
have a smaller search space, but it felt like a rabbit-hole and I wanted to
see it solve some harder puzzles! I started adding some more human-centric
constraints hoping one or two would be enough to prune the search space and
get across the finish line. Unfortunately the data didn't quite fit what I
needed to apply the constraints so I started v3. v3 relied on sets instead of
min/max range tuples and the data was made mostly immutable to avoid issues
with recursion. The mutable data is basically for pushing unit-level constraints 
to groups on the next iteration of Puzzle.update().

This version isn't particularly elegant, but it solves every puzzle I throw
at it in reasonable time (<1s for most puzzles, <5s for harder ones). It
captures most of the human-strategy and applies guess-and-test when the
human-strategies get stuck.

## Terminology
In the source I made up some names that probably aren't canonical
Picross/Nonogram terms.
* *group* - represents the solution to a single hint in a unit (row or column)
* *unit* - represents a row or column and contains the groups representing
  each hint in the unit 
* *span* - is a set of contiguous filled in squares
* *partition* - the space between X's

In future iterations I migth like to:
*  Revisit doing a simple+elegant version. For reference
   (https://github.com/seanluo1/Picross-Solver) looks pretty close to what I
   imagined it should be
* Implement it in a new language, probably Go since it's a common language at
  my current job. Possibly JS/Typescript for a web app version
* Try in a web app or mobile app to construct user-generated puzzles from
  pictures / emoji / doodles