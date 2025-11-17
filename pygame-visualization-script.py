# This code is intended to be run in a local Python environment with Pygame and h5py installed.
# It will not display a window in Google Colab.

import pygame
import numpy as np
import h5py # Import h5py
import time # Import time for potential delays
import sys

# Try to import tkinter for a GUI file chooser. If unavailable, we'll fall back to
# a command-line prompt or a default path.
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# Assuming width and height are defined globally or passed in
# If running this code separately, make sure width and height are defined.
# Example:
width = 400
height = 400
fps = 60 # Also ensure fps is defined

# Define colors for the states (0: empty, 1: rock, 2: paper, 3: scissors)
BLACK = (36, 36, 36)       # Empty
RED = (232, 87, 58)       # Rock
GREEN = (42, 163, 75)    # Paper
BLUE = (119, 86, 219)      # Scissors

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

# Choose a file: priority order -> sys.argv[1] -> tkinter chooser -> input prompt -> default path
output_filename = None
if len(sys.argv) > 1:
    output_filename = sys.argv[1]
else:
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        output_filename = filedialog.askopenfilename(initialdir='logs', title='Select simulation HDF5 file', filetypes=[('HDF5 files', '*.h5'), ('All files', '*.*')])
        root.destroy()
    else:
        try:
            prompt = "Enter path to HDF5 file (press Enter to use 'logs\\simulation_grids.h5'): "
            choice = input(prompt).strip()
            output_filename = choice if choice else 'logs\\simulation_grids.h5'
        except Exception:
            output_filename = 'logs\\simulation_grids.h5'

try:
    if not output_filename:
        raise FileNotFoundError('No file selected')

    with h5py.File(output_filename, 'r') as f:
        # Assuming datasets are named 'epoch_00000', 'epoch_00001', etc.
        # Get sorted list of epoch dataset names
        epoch_names = sorted([name for name in f.keys() if name.startswith('epoch_')])
        for name in epoch_names:
            grid_np = f[name][()] # Load the dataset as a NumPy array
            simulation_grids.append(grid_np)

    print(f"Successfully loaded {len(simulation_grids)} grid states from '{output_filename}'.")

except FileNotFoundError:
    print(f"Error: {output_filename} not found or not selected. Please run the simulation cell first or choose a valid file.")
    simulation_grids = [] # Empty list if file not found
except Exception as e:
    print(f"Error loading data from HDF5 file '{output_filename}': {e}")
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