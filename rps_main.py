import cupy as cp
import RockPaperScissors
import matplotlib.pyplot as plt
import seaborn as sns

'''
Here, we instatiate an RPS game and run it for a number of iterations. Then we visualize the game state after our iterations. Look in /utils/ for more scripts to manipulate games
'''

#Hyper-params:
k = 10000 #number of iterations

#Params:
p,q,gamma = 10, 10, 10 #We can pass in any floats, and we probabilities as the softmax p, g and gamma
width, height = 512,512
dims = [width,height] #dimension of our game
density = 0.5 #Initial starting density

p_settle, p_competition, p_mobility = (p/(p+q+gamma))(q/(p+q+gamma)),(gamma/(p+q+gamma))

#Initialize our game
game = RockPaperScissors(dims, density, [p_settle,p_competition,p_mobility])
game.seeding()

for i in range(k):
    game.update()
#Get our grid
grid = game.grid

#Boundary Entropy
entropy = game.get_entropy()
print(f"Our game's current boundary complexity is {entropy}")

numpy_grid = cp.asnumpy(game.grid)
# Use Seaborn to create a heatmap
plt.figure(figsize=(width/20, height/20)) # Adjust figure size based on grid dimensions
sns.heatmap(numpy_grid, cmap="viridis", cbar=False, square=True)
plt.title(f"State after {epochs} iterations (Seaborn)")
plt.axis('off') # Hide axes
plt.show()

