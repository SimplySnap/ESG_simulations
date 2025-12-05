import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import RockPaperScissors


epochs = 1950 #How many epochs do we want to visualize
width,height=512,512 #what size grid do we want
#we can also pass a (0,1] float density into RockPaperScissors's initialization

#Visualize a static frame after k iterations
game  = RockPaperScissors(dims=[width,height])
game.seeding()

for i in range(epochs):
  game.update()

numpy_grid = cp.asnumpy(game.grid)
# Use Seaborn to create a heatmap
plt.figure(figsize=(width/20, height/20)) # Adjust figure size based on grid dimensions
sns.heatmap(numpy_grid, cmap="viridis", cbar=False, square=True)
plt.title(f"State after {epochs} iterations (Seaborn)")
plt.axis('off') # Hide axes
plt.show()