from contextlib import nullcontext
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cupy as cp
import numpy as np # Import numpy
import gc
import RockPaperScissors


# Create an instance of the game and run a simulation to populate self.history
game_instance = None
gc.collect() #Collect garbage RAM
game_instance = RockPaperScissors()
epochs = 10000 # You can adjust the number of epochs for the static plot
game_instance.seeding()

# Append the initial counts after seeding to the history (transfer to CPU and convert to item)
initial_counts_cpu = [count.get().item() for count in game_instance.counts]
game_instance.history.append(initial_counts_cpu)

# Run the simulation with counting enabled to populate self.history
for epoch in range(epochs):
    game_instance.update(counting=True)

# Check if history was populated
if game_instance.history:
    # The history now contains lists of Python integers, no need for .get().item() again here
    # Convert the history data to a pandas DataFrame for plotting
    # Map the species numbers to strings for the 'Species' column
    species_map = {0: "Rock (1)", 1: "Paper (2)", 2: "Scissors (3)"}
    counts_df = pd.DataFrame(game_instance.history, columns=[species_map[i] for i in range(len(game_instance.species))])


    # Add an epoch number column
    counts_df['Epoch'] = range(len(counts_df))

    # Melt the DataFrame to a long format suitable for seaborn.lineplot
    counts_melted = counts_df.melt('Epoch', var_name='Species', value_name='Count')

    # Create the static seaborn line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=counts_melted, x='Epoch', y='Count', hue='Species')

    plt.title("Species Counts Over Time (Static Plot)")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
else:
    print("game_instance.history is empty. Ensure the simulation was run with counting enabled.")