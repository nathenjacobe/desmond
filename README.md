# A graphing calculator inspired by Desmos.

## Features:
- supports both explicit (of the form (y=) f(x)) and implicit relationships (of the form f(x,y) = g(x,y))
- all trigonometric functions: sin, cos, tan and their inverses and reciprocals, as well as the hyperbolic versions and their inverses and reciprocals
- permits implicit multiplication (i.e. you can do 2sin(x) instead of 2*sin(x), or sec(x)tan(x) instead of sec(x) * tan(x))
- draws asymptotes as red dashed lines for division-by-zero cases
- distributions: binomial(x, number_of_trials, probability), normal(x, mean, stdev), poisson(x, freq)
- constants: e, pi, euler-mascheroni (gamma; correct to 50 d.p.)
- generic: abs, floor, ceil, round, erf (approximated as tanh(xpi/sqrt(6))), factorial (uses Lanczos approximation for gamma function) and sqrt/cbrt

## To Do:
- add panning / zooming, ideally with caching to prevent expensive redraws
- fix single points being rendered radially for implicit
- fix lines being drawn with strange thicknesses for implicit
- allow multiple relations to be drawn simultaneously and coloured independantly, and allow them to be removed or hidden too.
- abandon python and rewrite this in rust so that i can manually implement CORDIC because it looks super cool
