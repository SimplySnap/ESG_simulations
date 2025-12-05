import cupy as cp


class RockPaperScissors():
  '''
  Initializes our Rock Paper Scissors game instance, with neighbor-sensitive rules
  This means, once some threshold (default 2) is reached, then a transition condition is checked,
  for every neighboring cell that imposes this transition condition

  Expects height, width, density and transition probabilities, where the probabilities are floats that some to one formatted as [p_settle, p_competition, p_mobility]

  We assign rock->1, paper->2, scissors->3
  '''

  # build grid
  def __init__(self, dims=[512,512], density=0.25, probs=[0.25,0.5,0.25]):
    self.width, self.height = dims
    self.grid = cp.zeros(dims, dtype=cp.int32) # Specify dtype
    self.species = [1,2,3] #How many species do we want in our system?
    self.counts = [0 for i in range(len(self.species))] # Array of population counts, for statistics later
    self.history = [] # Initialize history list
    self.density = density #change for different density initialization

    #transition probabilities
    self.p_settle, self.p_competition, self.p_mobility = probs

  def seeding(self):
    ''' Get starting positions of our grid '''
    density = self.density
    width,height = self.width,self.height
    # Sample values for the entire grid
    values = cp.random.choice([1, 2, 3], size=(width, height), p=[1/3, 1/3, 1/3]).astype(cp.int32) 
    toggles = cp.random.choice([True, False], size=(width, height), p=[density, 1-density])
    # Update the grid where toggle is True
    self.grid = cp.where(toggles, values, 0) # Apply values only where toggles is True

    # Count the number of each species after seeding
    for i, species_type in enumerate(self.species):
        self.counts[i] = cp.sum(self.grid == species_type)

  def get_entropy(self):
    '''
    Calculates the 'border complexity' or entropy of the system using CuPy broadcasting.
    This involves calculating global species proportions, constructing a lookup table for
    relation probabilities (q_values), and then computing the entropic contribution
    for all cell-neighbor pairs in parallel.

    Returns: scaled boundary entropy (cupy32 float)
    '''
    p_settle = self.p_settle
    p_competition = self.p_competition

    current_grid = self.grid
    current_height, current_width = current_grid.shape
    total_cells = current_grid.size

    # Calculate global species proportions (p(x))
    species_proportions = cp.zeros(4, dtype=cp.float32)
    species_proportions[0] = cp.sum(current_grid == 0) / total_cells # Empty cells
    species_proportions[1] = cp.sum(current_grid == 1) / total_cells # Rock
    species_proportions[2] = cp.sum(current_grid == 2) / total_cells # Paper
    species_proportions[3] = cp.sum(current_grid == 3) / total_cells # Scissors

    #Construct the q_lookup_table_log - efficient way to get q(x,y) contributions
    # Initialize with default competition log value for non-empty, different species interaction
    # this gives the minimal number of changes to our matrix
    q_lookup_table_log = cp.full((4, 4), cp.log2(1 - p_competition), dtype=cp.float32)

    #Set same species or both empty (t, t) to log2(1) = 0
    for t in range(4):
        q_lookup_table_log[t, t] = cp.log2(1.0)

    #Set empty cell (0, t) with non-empty neighbor to log2(1 - p_settle)
    for t in [1, 2, 3]:
        q_lookup_table_log[0, t] = cp.log2(1 - p_settle)

    #Set non-empty cell (t, 0) with empty neighbor to log2(1 - p_settle)
    for t in [1, 2, 3]:
        q_lookup_table_log[t, 0] = cp.log2(1 - p_settle)

    #Prepare current_grid and neighbors
    padded_grid = cp.pad(current_grid, 1, mode='wrap') # Use wrap to handle toroidal grid

    neighbors = cp.zeros((8, current_height, current_width), dtype=cp.int32)
    neighbors[0] = padded_grid[:-2, :-2] # Top-left
    neighbors[1] = padded_grid[:-2, 1:-1] # Top-center
    neighbors[2] = padded_grid[:-2, 2:]   # Top-right
    neighbors[3] = padded_grid[1:-1, :-2] # Middle-left
    neighbors[4] = padded_grid[1:-1, 2:]   # Middle-right
    neighbors[5] = padded_grid[2:, :-2]   # Bottom-left
    neighbors[6] = padded_grid[2:, 1:-1] # Bottom-center
    neighbors[7] = padded_grid[2:, 2:]   # Bottom-right

    #Compute entropic contribution using broadcasting
    current_grid_reshaped = current_grid[cp.newaxis, :, :]

    #Look up the cp.log2(q(x,y)) values for all cell-neighbor pairs
    log_q_values = q_lookup_table_log[current_grid_reshaped, neighbors]

    #Calculate the weight_grid_values for each cell based on its species type
    #species_proportions is 1D (length 4), current_grid is 2D (height x width)
    #This uses advanced indexing to map each cell's type to its proportion
    weight_grid_values = species_proportions[current_grid]

    #Compute the entropic contribution for each cell-neighbor pair
    #Reshape weight_grid_values for broadcasting with log_q_values (8, height, width)
    entropic_contributions = (1/8) * weight_grid_values[cp.newaxis, :, :] * log_q_values

    #Sum these contributions
    total_entropy = -cp.sum(entropic_contributions)

    #Apply the scaling factor
    scaled_total_entropy = total_entropy / cp.sqrt(current_height * current_width)#2D -> 1D

    #Return the final total_entropy
    return scaled_total_entropy



  def update(self, counting=False):
    '''
    Markov process where we update our grid based on starting probabilities.
    Integrates settlement, domination, and mobility as concurrent, mutually exclusive actions.
    Dominated cells become empty, mobility involves swapping with a neighbor.

    Returns: updated grid (cupy array)
    '''
    global p_settle, p_competition, p_mobility # Access global probability constants

    current_grid = self.grid
    current_height, current_width = current_grid.shape

    # Initialize new_grid with the current_grid state.
    # Cells not affected by any rule will retain their state.
    new_grid = cp.copy(current_grid)

    # 1. Pad the grid to handle neighbors at edges (for neighbor calculations)
    padded_grid = cp.pad(current_grid, 1, mode='wrap') # Use wrap to handle toroidal grid

    # 2. Get neighbor values for each cell in the current_grid
    #We store as an 8-depth 3D tensor so each cell has its neighbor dynamics easily caluclated!
    neighbors = cp.zeros((8, current_height, current_width), dtype=cp.int32)
    neighbors[0] = padded_grid[:-2, :-2] # Top-left
    neighbors[1] = padded_grid[:-2, 1:-1] # Top-center
    neighbors[2] = padded_grid[:-2, 2:]   # Top-right
    neighbors[3] = padded_grid[1:-1, :-2] # Middle-left
    neighbors[4] = padded_grid[1:-1, 2:]   # Middle-right
    neighbors[5] = padded_grid[2:, :-2]   # Bottom-left
    neighbors[6] = padded_grid[2:, 1:-1] # Bottom-center
    neighbors[7] = padded_grid[2:, 2:]   # Bottom-right

    # Count non-empty neighbors and neighbors of each type for each cell
    non_empty_neighbor_count = cp.sum(neighbors != 0, axis=0)
    rock_neighbor_counts = cp.sum(neighbors == 1, axis=0)
    paper_neighbor_counts = cp.sum(neighbors == 2, axis=0)
    scissors_neighbor_counts = cp.sum(neighbors == 3, axis=0)

    # 3. Generate independent random rolls for each action type
    rand_settle_roll = cp.random.rand(current_height, current_width)
    rand_dominate_roll = cp.random.rand(current_height, current_width)
    rand_mobility_roll = cp.random.rand(current_height, current_width)

    # 4. Calculate local probabilities for each action

    # Settlement probability (prob_settle_local)
    prob_settle_local = 1 - (1 - p_settle)**non_empty_neighbor_count
    # Ensure prob_settle_local is 0 where non_empty_neighbor_count is 0 (or where cell is not empty)
    prob_settle_local = cp.where(non_empty_neighbor_count == 0, 0.0, prob_settle_local)

    # Domination probability (prob_dominate_local)
    prob_dominate_local = cp.zeros((current_height, current_width), dtype=cp.float32)

    # Rock (1) dominated by Paper (2)
    rock_cells = (current_grid == 1)
    prob_dominate_local = cp.where(rock_cells, 1 - (1 - p_competition)**paper_neighbor_counts, prob_dominate_local)

    # Paper (2) dominated by Scissors (3)
    paper_cells = (current_grid == 2)
    prob_dominate_local = cp.where(paper_cells, 1 - (1 - p_competition)**scissors_neighbor_counts, prob_dominate_local)

    # Scissors (3) dominated by Rock (1)
    scissors_cells = (current_grid == 3)
    prob_dominate_local = cp.where(scissors_cells, 1 - (1 - p_competition)**rock_neighbor_counts, prob_dominate_local)

    # Mobility probability (prob_mobility_local)
    # Assuming max 8 neighbors for mobility consideration (Moore neighborhood)
    prob_mobility_local = 1 - (1 - p_mobility)**8
    # Only occupied cells can move
    prob_mobility_local = cp.where(current_grid == 0, 0.0, prob_mobility_local)

    # 5. Create masks for each action based on independent rolls and local probabilities, ensuring priority.
    # Priority: Settlement > Domination > Mobility

    # ----------------------- Settlement Mask ------------------------
    # Cells that roll for settlement AND are empty AND have non-empty neighbors.
    final_settle_mask = (current_grid == 0) & (rand_settle_roll < prob_settle_local)

    # ----------------------- Domination Mask ------------------------
    # Cells that roll for domination AND are occupied AND meet domination conditions
    # AND are not already settling.
    domination_condition_mask = \
        ((current_grid == 1) & (paper_neighbor_counts > 0)) | \
        ((current_grid == 2) & (scissors_neighbor_counts > 0)) | \
        ((current_grid == 3) & (rock_neighbor_counts > 0))

    final_dominate_mask = \
        (~final_settle_mask) & \
        (current_grid != 0) & \
        (rand_dominate_roll < prob_dominate_local) & \
        domination_condition_mask


    # ------------------------- Mobility Mask --------------------------
    # Cells that roll for mobility AND are occupied AND are not settling or dominating.
    mobility_condition_mask = (current_grid != 0)
    final_mobility_mask = \
        (~final_settle_mask) & \
        (~final_dominate_mask) & \
        (rand_mobility_roll < prob_mobility_local) & \
        mobility_condition_mask


    # 6. Apply actions to new_grid based on the final, mutually exclusive masks.

    # Apply Settlement: Empty cells adopt a random non-empty neighbor's species
    if cp.any(final_settle_mask):
        settling_y_coords, settling_x_coords = cp.where(final_settle_mask)
        num_settling_cells = settling_y_coords.shape[0]

        if num_settling_cells > 0:
            # For each settling cell, choose a random neighbor index (0-7)
            rand_neighbor_idx_for_settlers = cp.random.randint(0, 8, size=num_settling_cells)

            # Get the species of these randomly chosen neighbors
            # `neighbors[k, y, x]` gives the k-th neighbor's species for cell (y,x)
            chosen_neighbor_species = neighbors[rand_neighbor_idx_for_settlers, settling_y_coords, settling_x_coords]

            # Only settle if the chosen neighbor is not empty (species != 0)
            valid_settlement_mask_for_chosen_neighbor = (chosen_neighbor_species != 0)

            # Apply the species of the valid chosen neighbors to the new_grid
            new_grid[settling_y_coords[valid_settlement_mask_for_chosen_neighbor],
                     settling_x_coords[valid_settlement_mask_for_chosen_neighbor]] = \
                chosen_neighbor_species[valid_settlement_mask_for_chosen_neighbor]

    # Apply Domination: Dominated cells become empty (0)
    if cp.any(final_dominate_mask):
        new_grid[final_dominate_mask] = 0

    # Apply Mobility: Occupied cells swap their state with a randomly chosen neighbor
    mover_y, mover_x = cp.where(final_mobility_mask)
    num_movers = mover_y.shape[0]

    if num_movers > 0:
        # Select a random neighbor for each mover (index 0-7 for 8 neighbors)
        rand_neighbor_idx = cp.random.randint(0, 8, size=num_movers)

        # Define offsets for 8 neighbors (Moore neighborhood)
        offsets_y = cp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=cp.int32)
        offsets_x = cp.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=cp.int32)

        # Calculate target coordinates for each mover, ensuring wrap-around boundaries
        target_y = (mover_y + offsets_y[rand_neighbor_idx]) % current_height
        target_x = (mover_x + offsets_x[rand_neighbor_idx]) % current_width

        # To perform the swap in parallel without race conditions or overwriting intermediate values:
        # We need to read values from `current_grid` (the state *before* any updates this tick)
        # and write them to appropriate locations in `new_grid`.

        # Get the current values at the mover's original position and their target neighbor's position
        values_at_mover_positions = current_grid[mover_y, mover_x]
        values_at_target_positions = current_grid[target_y, target_x]

        # Assign the target neighbor's value to the mover's original position in new_grid
        new_grid[mover_y, mover_x] = values_at_target_positions

        # Assign the mover's original value to the target neighbor's position in new_grid
        # Note: If multiple movers target the same cell, the last write will prevails.
        # This is generally acceptable for this type of simulation as order is not strictly critical.
        new_grid[target_y, target_x] = values_at_mover_positions

    self.grid = new_grid # Update the grid to the new state

    # Update counts if counting is enabled
    if counting:
      for i, species_type in enumerate(self.species):
          self.counts[i] = cp.sum(self.grid == species_type)
      self.history.append([count.get().item() for count in self.counts]) #Add counts to history (transfer to CPU for plotting)

    return self.grid