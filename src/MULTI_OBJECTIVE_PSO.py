import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# objective functions
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum((x - np.sqrt(3)/3)**2)

def evaluate_objectives(x):
    return np.array([f1(x), f2(x)])

class MPSO:
    def __init__(self, num_particles, num_dimensions, bounds, archive_size=100, w_max=0.9, w_min=0.4):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.archive_size = archive_size
        
        # PSO parameters
        self.w_max = w_max  
        self.w_min = w_min  
        self.c1 = 2.0  
        self.c2 = 2.0  
        self.c3 = 1.5  # Local best influence
        
        # Initialize particles and velocities
        self.positions = np.random.uniform(bounds[0], bounds[1], 
                                         (num_particles, num_dimensions))
        self.velocities = np.zeros((num_particles, num_dimensions))
        
        # Initialize personal best
        self.pbest = self.positions.copy()
        self.pbest_objectives = np.array([evaluate_objectives(p) for p in self.pbest])
        
        # Initialize archive for non-dominated solutions
        self.archive_positions = []
        self.archive_objectives = []
        
        # Initialize tracking variables for convergence analysis
        self.global_objective_history = []

    def dominates(self, obj1, obj2):
        """Return True if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def update_archive(self, position, objectives):
        """Update archive with new non-dominated solution"""
        dominated = False
        
        # Check if the new solution is dominated by any archive solution
        i = 0
        while i < len(self.archive_objectives):
            if self.dominates(self.archive_objectives[i], objectives):
                dominated = True
                break
            elif self.dominates(objectives, self.archive_objectives[i]):
                del self.archive_positions[i]
                del self.archive_objectives[i]
                i -= 1
            i += 1
        
        # Add new solution if not dominated
        if not dominated:
            self.archive_positions.append(position.copy())
            self.archive_objectives.append(objectives.copy())
        
        # Maintain archive size using crowding distance
        if len(self.archive_positions) > self.archive_size:
            self.reduce_archive()
    
    def reduce_archive(self):
        """Reduce archive size using crowding distance"""
        objectives = np.array(self.archive_objectives)
        
        # Calculate crowding distance
        crowd_dist = np.zeros(len(objectives))
        
        for m in range(objectives.shape[1]):
            # Sort by each objective
            idx = objectives[:, m].argsort()
            sorted_objectives = objectives[idx]
            
            # Set boundary points to infinity
            crowd_dist[idx[0]] = np.inf
            crowd_dist[idx[-1]] = np.inf
            
            # Calculate crowding distance for other points
            obj_range = objectives[:, m].max() - objectives[:, m].min()
            if obj_range > 0:
                for i in range(1, len(idx)-1):
                    crowd_dist[idx[i]] += (sorted_objectives[i+1, m] - 
                                         sorted_objectives[i-1, m]) / obj_range
        
        # Remove solutions with smallest crowding distance
        while len(self.archive_positions) > self.archive_size:
            idx = crowd_dist.argmin()
            del self.archive_positions[idx]
            del self.archive_objectives[idx]
            crowd_dist = np.delete(crowd_dist, idx)
    
    def select_leader(self):
        """Select leader from archive using binary tournament"""
        if not self.archive_positions:
            return self.positions[np.random.randint(self.num_particles)]
        
        idx1, idx2 = np.random.randint(0, len(self.archive_positions), 2)
        if np.random.random() < 0.5:
            return self.archive_positions[idx1]
        return self.archive_positions[idx2]
    
    def find_local_best(self, particle_idx):
        """Find the local best solution by comparing nearby particles"""
        neighbor_idx = np.random.choice(np.arange(self.num_particles), size=3, replace=False)
        local_best_idx = neighbor_idx[0]
        for idx in neighbor_idx:
            if self.dominates(self.pbest_objectives[idx], self.pbest_objectives[local_best_idx]):
                local_best_idx = idx
        return self.pbest[local_best_idx]
    
    def optimize(self, max_iter):
        for iteration in range(max_iter):
            # Update inertia weight
            self.w = self.w_max - ((self.w_max - self.w_min) * iteration / max_iter)
            
            for i in range(self.num_particles):
                # Evaluate current position
                current_objectives = evaluate_objectives(self.positions[i])
                
                # Update personal best
                if self.dominates(current_objectives, self.pbest_objectives[i]):
                    self.pbest[i] = self.positions[i].copy()
                    self.pbest_objectives[i] = current_objectives.copy()
                
                # Update archive
                self.update_archive(self.positions[i], current_objectives)
                
                # Select leader and local best
                leader = self.select_leader()
                local_best = self.find_local_best(i)
                
                # Update velocity and position
                r1, r2, r3 = np.random.rand(3)
                self.velocities[i] = (self.w * self.velocities[i] + 
                                      self.c1 * r1 * (self.pbest[i] - self.positions[i]) +
                                      self.c2 * r2 * (leader - self.positions[i]) +
                                      self.c3 * r3 * (local_best - self.positions[i]))
                
                self.positions[i] += self.velocities[i]
                
                # Apply bounds
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
            
            # Track global objective value for convergence
            archive_objectives = np.array(self.archive_objectives)
            global_obj_value = archive_objectives.sum(axis=1).min() if len(archive_objectives) > 0 else np.inf
            self.global_objective_history.append(global_obj_value)

        print("\nOptimization Complete")

# Main execution
if __name__ == "__main__":
    # Parameters
    num_particles = 100
    num_dimensions = 3
    bounds = (0, 2)
    max_iter = 100
    
    # Create and run MPSO
    mpso = MPSO(num_particles, num_dimensions, bounds)
    mpso.optimize(max_iter)
    
    # Plot Pareto front
    archive_objectives = np.array(mpso.archive_objectives)
    plt.figure(figsize=(10, 8))
    plt.scatter(archive_objectives[:, 0], archive_objectives[:, 1], 
                c='blue', label='Pareto front')
    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.title('Pareto Front of Non-dominated Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(mpso.global_objective_history, label='Global Objective')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Value')
    plt.title('Convergence Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
