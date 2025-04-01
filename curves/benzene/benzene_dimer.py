import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def generate_benzene_coordinates():
    """
    Generate coordinates for a single benzene molecule in the xy-plane.
    
    Returns:
    numpy.ndarray: Array of shape (12, 3) containing the coordinates of all atoms
                  (6 Carbon atoms and 6 Hydrogen atoms)
    """
    # Fixed parameters
    C_C_bond_length = 1.39  # Angstrom
    C_H_bond_length = 1.09  # Angstrom
    
    # Calculate positions of carbon atoms in a regular hexagon in the xy-plane
    carbon_positions = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = C_C_bond_length * np.cos(angle)
        y = C_C_bond_length * np.sin(angle)
        z = 0.0
        carbon_positions.append(np.array([x, y, z]))
    
    # Calculate positions of hydrogen atoms
    hydrogen_positions = []
    for i in range(6):
        # Vector from origin to carbon
        c_vec = carbon_positions[i]
        c_vec_unit = c_vec / np.linalg.norm(c_vec)
        
        # Hydrogen position is further out in the same direction
        h_pos = carbon_positions[i] + c_vec_unit * C_H_bond_length
        hydrogen_positions.append(h_pos)
    
    # Combine carbon and hydrogen positions
    return np.vstack([carbon_positions, hydrogen_positions])

def calculate_center_of_mass(coordinates):
    """
    Calculate the center of mass of a molecule.
    Assuming C and H masses (simplified).
    
    Parameters:
    coordinates (numpy.ndarray): Atom coordinates
    
    Returns:
    numpy.ndarray: Center of mass coordinates
    """
    # Atomic masses (amu)
    masses = {
        'C': 12.01,
        'H': 1.01
    }
    
    # For benzene, first 6 atoms are carbon, next 6 are hydrogen
    atom_masses = np.array([masses['C']] * 6 + [masses['H']] * 6)
    
    # Calculate center of mass
    total_mass = np.sum(atom_masses)
    weighted_coords = coordinates * atom_masses[:, np.newaxis]
    com = np.sum(weighted_coords, axis=0) / total_mass
    
    return com

def generate_benzene_dimer_coordinates(distance):
    """
    Generate coordinates for a benzene dimer with specified center-to-center distance.
    First benzene is in the xy-plane. Second benzene is stacked above it.
    
    Parameters:
    distance (float): Center-to-center distance in Angstroms
    
    Returns:
    numpy.ndarray: Array of shape (24, 3) containing the coordinates of all atoms
    """
    # Generate coordinates for the first benzene (in xy-plane)
    benzene1 = generate_benzene_coordinates()
    
    # Calculate center of mass for the first benzene
    com1 = calculate_center_of_mass(benzene1)
    
    # Shift the first benzene so its center of mass is at the origin
    benzene1 = benzene1 - com1
    
    # Make a copy of benzene1 for the second molecule
    benzene2 = benzene1.copy()
    
    # Position the second benzene above the first, shifted by the specified distance in z-direction
    benzene2[:, 2] += distance
    
    # Combine both benzenes
    return np.vstack([benzene1, benzene2])

def calculate_energy(distance):
    """
    Calculate a model energy for benzene dimer dissociation.
    This uses a simplified model for pi-pi stacking interactions.
    
    Parameters:
    distance (float): Center-to-center distance in Angstroms
    
    Returns:
    float: Energy in arbitrary units
    """
    # Parameters for a simple Lennard-Jones-like potential
    epsilon = 10.0  # Depth of potential well
    sigma = 3.8     # Optimal stacking distance
    
    # Simplified potential for pi-pi stacking
    energy = epsilon * ((sigma/distance)**12 - 2*(sigma/distance)**6)
    
    return energy

def write_xyz_file(filename, coordinates, distance):
    """
    Write the coordinates to an XYZ file.
    
    Parameters:
    filename (str): Name of the output file
    coordinates (numpy.ndarray): Array containing atom coordinates
    distance (float): Center-to-center distance for the file comment line
    """
    # First 6 atoms of each benzene are carbon, next 6 are hydrogen
    atoms = ['C'] * 6 + ['H'] * 6 + ['C'] * 6 + ['H'] * 6
    
    with open(filename, 'w') as f:
        f.write(f'{len(atoms)}\n')
        f.write(f'Benzene dimer with center-to-center distance = {distance:.2f} Angstrom\n')
        
        for atom, pos in zip(atoms, coordinates):
            f.write(f'{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')

# Main script
if __name__ == "__main__":
    # Distances to simulate (Angstrom)
    start_distance = 3.5  # Typical pi-stacking distance
    end_distance = 14.0
    num_steps = 50
    
    distances = np.linspace(start_distance, end_distance, num_steps)
    
    # Calculate energies
    energies = [calculate_energy(d) for d in distances]
    
    # Write XYZ files for each step
    for i, distance in enumerate(distances):
        coordinates = generate_benzene_dimer_coordinates(distance)
        # write_xyz_file(f'benzene_dimer_{i+1:03d}.xyz', coordinates, distance)
        write_xyz_file(f'benzene_dimer_{distance:.2f}.xyz', coordinates, distance)
    
    # Create a merged XYZ file for animation in visualization software
    with open('benzene_dimer_animation.xyz', 'w') as outfile:
        for i, distance in enumerate(distances):
            coordinates = generate_benzene_dimer_coordinates(distance)
            atoms = ['C'] * 6 + ['H'] * 6 + ['C'] * 6 + ['H'] * 6
            
            outfile.write(f'{len(atoms)}\n')
            outfile.write(f'Benzene dimer with center-to-center distance = {distance:.2f} Angstrom\n')
            
            for atom, pos in zip(atoms, coordinates):
                outfile.write(f'{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')
    
    # Create an energy plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, 'b-', linewidth=2)
    plt.xlabel('Center-to-Center Distance (Å)')
    plt.ylabel('Energy (arbitrary units)')
    plt.title('Benzene Dimer Dissociation Energy')
    plt.grid(True)
    plt.savefig('benzene_dimer_energy.png', dpi=300)
    
    # Create a 3D animation of the dissociation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize scatter plots for carbon and hydrogen atoms
    carbon_scatter = ax.scatter([], [], [], c='black', s=100, label='Carbon')
    hydrogen_scatter = ax.scatter([], [], [], c='lightblue', s=50, label='Hydrogen')
    
    # Initialize lines for bonds
    # For benzene, we need 12 C-C bonds (6 for each molecule) + 12 C-H bonds (6 for each molecule)
    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0] for _ in range(24)]
    
    # Set axis labels
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Function to initialize the animation
    def init():
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-2, 12)
        ax.legend()
        return (carbon_scatter, hydrogen_scatter) + tuple(lines)
    
    def get_bonds(coordinates):
        """
        Get the bonds between atoms for visualization.
        
        Parameters:
        coordinates (numpy.ndarray): Atom coordinates
        
        Returns:
        list: List of pairs of indices representing bonds
        """
        bonds = []
        # C-C bonds for first benzene (indices 0-5)
        for i in range(5):
            bonds.append((i, i+1))
        bonds.append((5, 0))  # Close the ring
        
        # C-H bonds for first benzene
        for i in range(6):
            bonds.append((i, i+6))
        
        # C-C bonds for second benzene (indices 12-17)
        for i in range(12, 17):
            bonds.append((i, i+1))
        bonds.append((17, 12))  # Close the ring
        
        # C-H bonds for second benzene
        for i in range(6):
            bonds.append((i+12, i+18))
        
        return bonds
    
    # Function to update the animation at each frame
    def update(frame):
        coordinates = generate_benzene_dimer_coordinates(distances[frame])
        
        # Update positions of carbon atoms (indices 0-5 and 12-17)
        carbon_pos = np.vstack([coordinates[:6], coordinates[12:18]])
        carbon_scatter._offsets3d = (carbon_pos[:, 0], carbon_pos[:, 1], carbon_pos[:, 2])
        
        # Update positions of hydrogen atoms (indices 6-11 and 18-23)
        hydrogen_pos = np.vstack([coordinates[6:12], coordinates[18:24]])
        hydrogen_scatter._offsets3d = (hydrogen_pos[:, 0], hydrogen_pos[:, 1], hydrogen_pos[:, 2])
        
        # Update bond lines
        bonds = get_bonds(coordinates)
        for i, (idx1, idx2) in enumerate(bonds):
            lines[i].set_data([coordinates[idx1, 0], coordinates[idx2, 0]], 
                              [coordinates[idx1, 1], coordinates[idx2, 1]])
            lines[i].set_3d_properties([coordinates[idx1, 2], coordinates[idx2, 2]])
        
        ax.set_title(f'Benzene Dimer Dissociation: Distance = {distances[frame]:.2f} Å')
        
        return (carbon_scatter, hydrogen_scatter) + tuple(lines)
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(distances), init_func=init,
                       interval=200, blit=False)
    
    # Save the animation (requires ffmpeg)
    try:
        ani.save('benzene_dimer_dissociation.mp4', writer='ffmpeg', fps=10, dpi=150)
        print("Animation saved as 'benzene_dimer_dissociation.mp4'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Displaying animation in interactive window instead.")
        plt.show()
    
    print("Simulation complete!")
    print(f"- XYZ files generated for {num_steps} distances from {start_distance} to {end_distance} Angstroms")
    print("- Combined XYZ animation file saved as 'benzene_dimer_animation.xyz'")
    print("- Energy plot saved as 'benzene_dimer_energy.png'")
