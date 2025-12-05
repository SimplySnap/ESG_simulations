import cupy as cp
import h5py
import RockPaperScissors

'''
For storing simulations of our RPS game. We store as an HDF5 file and visualize in pygame
'''


epochs = 4000 #number of epochs
save_interval = 5 #sve every 1000 epochs
dims = [512,512] #width, height

# Initialize - can also pass in density and transition probabilities
game = RockPaperScissors(width=dims)
game.seeding()

# Create an HDF5 file to store the grids
output_filename = 'simulation_grids.h5'

with h5py.File(output_filename, 'w') as f:
    for i in range(epochs):
        game.update()
        if i % save_interval == 0:
            # Convert CuPy grid to NumPy array
            grid_np = cp.asnumpy(game.grid)
            # Create a dataset in the HDF5 file for the current epoch
            f.create_dataset(f'epoch_{i:05d}', data=grid_np)

print(f"Simulation finished. Saved {epochs / save_interval} grid states to '{output_filename}' every {save_interval} epochs.")