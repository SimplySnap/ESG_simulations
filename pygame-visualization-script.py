# This code is intended to be run in a local Python environment with Pygame and h5py installed.
# It will not display a window in Google Colab.

import pygame
import numpy as np
import h5py # Import h5py
import time # Import time for potential delays

# Assuming width and height are defined globally or passed in
# If running this code separately, make sure width and height are defined.
# Example:
width = 800
height = 400
fps = 30 # Also ensure fps is defined

# Define colors for the states (0: empty, 1: rock, 2: paper, 3: scissors)
BLACK = (0, 0, 0)       # Empty
RED = (255, 0, 0)       # Rock
GREEN = (0, 255, 0)     # Paper
BLUE = (0, 0, 255)      # Scissors

COLORS = {
    0: BLACK,
    1: RED,
    2: GREEN,
    3: BLUE
}

# Initialize Pygame
pygame.init()

# Set the dimensions of the window
screen_width = width
screen_height = height
screen = pygame.display.set_mode((screen_width, screen_height))

# Set the window title
pygame.display.set_caption("Rock Paper Scissors Simulation")

# Load the simulation log from the HDF5 file
simulation_grids = []
try:
    output_filename = 'logs\simulation_grids.h5' # Define the filename
    with h5py.File(output_filename, 'r') as f:
        # Assuming datasets are named 'epoch_00000', 'epoch_00001', etc.
        # Get sorted list of epoch dataset names
        epoch_names = sorted([name for name in f.keys() if name.startswith('epoch_')])
        for name in epoch_names:
            grid_np = f[name][()] # Load the dataset as a NumPy array
            simulation_grids.append(grid_np)

    print(f"Successfully loaded {len(simulation_grids)} grid states from '{output_filename}'.")

except FileNotFoundError:
    print(f"Error: {output_filename} not found. Please run the simulation cell first.")
    simulation_grids = [] # Empty list if file not found
except Exception as e:
    print(f"Error loading data from HDF5 file: {e}")
    simulation_grids = []


# --- Drawing parameters ---
# Calculate cell size
cell_size_x = screen_width // width
cell_size_y = screen_height // height

# Main game loop
running = True
clock = pygame.time.Clock()
current_epoch_index = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if simulation_grids and current_epoch_index < len(simulation_grids):
        current_grid = simulation_grids[current_epoch_index] # Corrected: Access the grid directly from the list

        # Clear the screen
        screen.fill(BLACK)

        # Draw the grid
        for y in range(height):
            for x in range(width):
                state = current_grid[x,y]
                color = COLORS.get(state, BLACK) # Default to black if state is unknown
                pygame.draw.rect(screen, color, (x * cell_size_x, y * cell_size_y, cell_size_x, cell_size_y))

        # Update the display
        pygame.display.flip()

        # Move to the next epoch
        current_epoch_index += 1

        # Control playback speed
        clock.tick(fps) # Use the fps constant defined earlier
        if current_epoch_index%10 == 0:
            print(f"Epoch {current_epoch_index} passed")

    elif simulation_grids:
        # Stop or loop when simulation ends
        print("Simulation visualization finished.")
        running = False # Stop after the last frame

    else:
        # If no grids were loaded, exit
        running = False


# Quit Pygame
pygame.quit()