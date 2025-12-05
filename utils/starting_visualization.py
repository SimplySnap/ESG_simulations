import matplotlib.pyplot as plt
import cupy as cp
import seaborn as sns
import RockPaperScissors


# Create an instance of the game
width,height = 512,512 # params we can change for different dimensions
#can also pass in density set as some float, and transition probabilities as a list of floats summing to 1

game = RockPaperScissors(dims=(width,height))
game.seeding()
#Print number of instances (species)
print(game.counts)

# Convert the CuPy grid to a NumPy array
numpy_grid = cp.asnumpy(game.grid)

# Use Seaborn to create a heatmap
plt.figure(figsize=(width/20, height/20)) # Adjust figure size based on grid dimensions
sns.heatmap(numpy_grid, cmap="viridis", cbar=False, square=True)
plt.title("Initial Grid State (Seaborn)")
plt.axis('off') # Hide axes
plt.show()