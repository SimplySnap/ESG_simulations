# Parallel Evolutionary Spatial Cyclic Games
An evolutionary game is an application of game theory to evolving populations in biology, where multiple ‘strategies’ (or species) face off against each other. 
In this repo, we use CuPy to simulate basic environmental games that exhibit spatial cyclic dominance.

![Example Game State](pics/0mob-500-2.png)

We also derive and investigate an entropy heuristic we dub 'boundary complexity':

$$C:=-\sum_x\sum_{y \in N(x)} p(x)\ln(q(x,y))$$

where $p(x)$ measures the population ratio of strategy 'x', and $/lnq(x,y)$ measures the surprise of a strategy $x$ being adjacent to a neighboring strategy $y$. 
This measures the 'complexity' of our overall system and can be applied in any multi-state discrete spatial system.

For more information, view section 4 of the pdf written report.


----- File information -----

rps_main.py : our main file that runs a game for a number of iterations and outputs a static visualization of the system

RockPaperScissors.py : our game class, that we call in other functions

pygame-visualization-script.py : visualizing a dynamic game system in pygame

utils : various scripts for saving game states, trajectories and statistical visualizations
