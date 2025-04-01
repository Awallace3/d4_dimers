import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def generate_water_dimer_coordinates(distance):
    """
    Generate coordinates for a water dimer with specified O-O distance.
    First water molecule is placed in the xy-plane.
    Second water molecule is positioned to form a hydrogen bond with the first.
    
    Parameters:
    distance (float): O-O distance in Angstroms
    
    Returns:
    numpy.ndarray: Array of shape (6, 3) containing the coordinates of all atoms
    """
    # Fixed parameters
    OH_bond_length = 0.96  # Angstrom
    HOH_angle = 104.5 * np.pi / 180  # Convert to radians
    
    # First water molecule (in xy-plane)
    O1_pos = np.array([0.0, 0.0, 0.0])
    H1a_pos = np.array([-OH_bond_length, 0.0, 0.0])
    
    # Calculate position of the second hydrogen to keep the molecule in xy-plane
    H1b_pos = np.array([-OH_bond_length * np.cos(HOH_angle), 
                         -OH_bond_length * np.sin(HOH_angle), 
                         0.0])
    
    # Second water molecule
    # Position O2 at the specified distance along x-axis from O1
    H2a_pos = np.array([distance, 0.0, 0.0])
    O2_pos = H2a_pos + np.array([OH_bond_length, 0.0, 0.0])
    
    # Calculate positions for H2a (hydrogen facing O1 to form the hydrogen bond)
    # and H2b (the other hydrogen)
    # H2a_pos = O2_pos + np.array([-OH_bond_length, 0.0, 0.0])
    
    # Position H2b to maintain the HOH angle
    H2b_pos = O2_pos + np.array([-OH_bond_length * np.cos(HOH_angle),
                                 -OH_bond_length * np.sin(HOH_angle),
                                 0.0])
    
    # Combine all positions
    coordinates = np.vstack([O1_pos, H1a_pos, H1b_pos, O2_pos, H2a_pos, H2b_pos])
    
    return coordinates

def calculate_energy(distance):
    """
    Calculate a simple model energy for water dimer dissociation.
    This is a very simplified model using a Lennard-Jones-like potential.
    
    Parameters:
    distance (float): O-O distance in Angstroms
    
    Returns:
    float: Energy in arbitrary units
    """
    epsilon = 5.0  # Depth of potential well
    sigma = 3.0    # Distance at which potential is zero
    
    # Simplified Lennard-Jones potential
    energy = epsilon * ((sigma/distance)**12 - 2*(sigma/distance)**6)
    
    return energy

def write_xyz_file(filename, coordinates, distance):
    """
    Write the coordinates to an XYZ file.
    
    Parameters:
    filename (str): Name of the output file
    coordinates (numpy.ndarray): Array of shape (6, 3) containing the coordinates
    distance (float): O-O distance for the file comment line
    """
    atoms = ['O', 'H', 'H', 'O', 'H', 'H']
    
    with open(filename, 'w') as f:
        f.write('6\n')
        f.write(f'Water dimer with O-O distance = {distance:.2f} Angstrom\n')
        
        for atom, pos in zip(atoms, coordinates):
            f.write(f'{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')

# Main script
if __name__ == "__main__":
    # Distances to simulate (Angstrom)
    start_distance = 2.0
    end_distance = 14.0
    num_steps = 50
    
    distances = np.linspace(start_distance, end_distance, num_steps)
    
    # Calculate energies
    energies = [calculate_energy(d) for d in distances]
    
    # Write XYZ files for each step
    for i, distance in enumerate(distances):
        coordinates = generate_water_dimer_coordinates(distance)
        write_xyz_file(f'water_dimer_{distance:.2f}.xyz', coordinates, distance)
    
    # Create a merged XYZ file for animation in visualization software
    with open('water_dimer_animation.xyz', 'w') as outfile:
        for i, distance in enumerate(distances):
            coordinates = generate_water_dimer_coordinates(distance)
            atoms = ['O', 'H', 'H', 'O', 'H', 'H']
            
            outfile.write('6\n')
            outfile.write(f'Water dimer with O-O distance = {distance:.2f} Angstrom\n')
            
            for atom, pos in zip(atoms, coordinates):
                outfile.write(f'{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')
    
    # Create an energy plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, 'b-', linewidth=2)
    plt.xlabel('O-O Distance (Å)')
    plt.ylabel('Energy (arbitrary units)')
    plt.title('Water Dimer Dissociation Energy')
    plt.grid(True)
    plt.savefig('water_dimer_energy.png', dpi=300)
    
    # Create a 3D animation of the dissociation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize scatter plots for oxygen and hydrogen atoms
    oxygen_scatter = ax.scatter([], [], [], c='red', s=100, label='Oxygen')
    hydrogen_scatter = ax.scatter([], [], [], c='lightblue', s=50, label='Hydrogen')
    
    # Initialize lines for bonds
    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0] for _ in range(5)]
    
    # Set axis labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Function to initialize the animation
    def init():
        ax.set_xlim(-3, 13)
        ax.set_ylim(-3, 13)
        ax.set_zlim(-3, 13)
        ax.legend()
        return (oxygen_scatter, hydrogen_scatter) + tuple(lines)
    
    # Function to update the animation at each frame
    def update(frame):
        coordinates = generate_water_dimer_coordinates(distances[frame])
        
        # Update positions of oxygen atoms (indices 0 and 3)
        oxygen_pos = coordinates[[0, 3]]
        oxygen_scatter._offsets3d = (oxygen_pos[:, 0], oxygen_pos[:, 1], oxygen_pos[:, 2])
        
        # Update positions of hydrogen atoms (indices 1, 2, 4, 5)
        hydrogen_pos = coordinates[[1, 2, 4, 5]]
        hydrogen_scatter._offsets3d = (hydrogen_pos[:, 0], hydrogen_pos[:, 1], hydrogen_pos[:, 2])
        
        # Update bond lines
        # O1-H1a bond
        lines[0].set_data([coordinates[0, 0], coordinates[1, 0]], [coordinates[0, 1], coordinates[1, 1]])
        lines[0].set_3d_properties([coordinates[0, 2], coordinates[1, 2]])
        
        # O1-H1b bond
        lines[1].set_data([coordinates[0, 0], coordinates[2, 0]], [coordinates[0, 1], coordinates[2, 1]])
        lines[1].set_3d_properties([coordinates[0, 2], coordinates[2, 2]])
        
        # H2a-O2 bond
        lines[2].set_data([coordinates[4, 0], coordinates[3, 0]], [coordinates[4, 1], coordinates[3, 1]])
        lines[2].set_3d_properties([coordinates[4, 2], coordinates[3, 2]])
        
        # H2b-O2 bond
        lines[3].set_data([coordinates[5, 0], coordinates[3, 0]], [coordinates[5, 1], coordinates[3, 1]])
        lines[3].set_3d_properties([coordinates[5, 2], coordinates[3, 2]])
        
        # H-bond (dashed line between O1 and H2a)
        if distances[frame] < 4.0:  # Only show H-bond for close distances
            lines[4].set_data([coordinates[0, 0], coordinates[4, 0]], [coordinates[0, 1], coordinates[4, 1]])
            lines[4].set_3d_properties([coordinates[0, 2], coordinates[4, 2]])
            lines[4].set_linestyle('--')
        else:
            lines[4].set_data([], [])
            lines[4].set_3d_properties([])
        
        ax.set_title(f'Water Dimer Dissociation: O-O Distance = {distances[frame]:.2f} Å')
        
        return (oxygen_scatter, hydrogen_scatter) + tuple(lines)
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(distances), init_func=init,
                       interval=200, blit=False)
    
    # Save the animation (requires ffmpeg)
    try:
        ani.save('water_dimer_dissociation.mp4', writer='ffmpeg', fps=10, dpi=150)
        print("Animation saved as 'water_dimer_dissociation.mp4'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Displaying animation in interactive window instead.")
        plt.show()
    
    print("Simulation complete!")
    print(f"- XYZ files generated for {num_steps} distances from {start_distance} to {end_distance} Angstroms")
    print("- Combined XYZ animation file saved as 'water_dimer_animation.xyz'")
    print("- Energy plot saved as 'water_dimer_energy.png'")
