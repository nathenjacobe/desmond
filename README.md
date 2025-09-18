# A graphing calculator inspired by Desmos.

https://github.com/user-attachments/assets/baba73c2-7a68-4a53-864e-3796bbec23d7
Yes, these graphs are (more or less) what Desmos shows as well.

## Features:
- supports both explicit (of the form (y=) f(x)) and implicit relationships (of the form f(x,y) = g(x,y)) using marching squares.
- all trigonometric functions: sin, cos, tan and their reciprocals, inverses and inverse reciprocals, as well as the hyperbolic versions and their reciprocals, inverses and inverse reciprocals.
- permits implicit multiplication (i.e. you can do 2sin(x) instead of 2*sin(x), or sec(x)tan(x) instead of sec(x) * tan(x))
- draws asymptotes as red dashed lines for division-by-zero cases
- distributions: binomial(x, number_of_trials, probability), normal(x, mean, stdev), poisson(x, freq), geo(x, probability).
- constants: e, pi, euler-mascheroni (gamma; correct to 50 d.p.)
- generic: abs, floor, ceil, round, max, min, clamp, erf (approximated as tanh(xpi/sqrt(6))), factorial (uses Lanczos approximation for gamma function), log and sqrt/cbrt

## To Do:
- add panning / zooming, ideally with caching to prevent expensive redraws
- add implicit brackets (i.e. sinx instead of sin(x)) - naively add brackets to the end of the LHS / RHS if incomplete.
- fix asymptotes especially at lower resolutions. A common problem is that the asymptotes get only rendered partially because the gradient gets so steep.
- abandon python and rewrite this in rust so that i can manually implement CORDIC because it looks super cool, even if my implementation would be slower than native.

