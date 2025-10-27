import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class ImprovedParticleSwarmOptimizer:
    def __init__(self, objective_function, bounds, num_particles=50, max_iter=150,
                 w_max=0.9, w_min=0.4, c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5,
                 v_max_coef=0.1, n_restarts=3):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.dim = len(bounds)
        self.w_max = w_max
        self.w_min = w_min
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.v_max_coef = v_max_coef
        self.n_restarts = n_restarts
        
        self.v_max = np.array([self.v_max_coef * (b[1] - b[0]) for b in bounds])
        
        self.global_best_position = None
        self.global_best_value = float('inf')
        
        self.convergence = []
        self.computation_time = 0
        self.all_runs_convergence = []
        self.all_runs_best_positions = []
        self.all_runs_best_values = []

    def initialize_particles(self):
        positions = np.random.uniform([b[0] for b in self.bounds], [b[1] for b in self.bounds], (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.num_particles, self.dim))
        return positions, velocities

    def optimize_single_run(self, run_number):
        positions, velocities = self.initialize_particles()
        
        p_best_positions = positions.copy()
        p_best_values = np.array([self.objective_function(*p) for p in positions])
        
        run_best_position = p_best_positions[np.argmin(p_best_values)]
        run_best_value = np.min(p_best_values)
        
        if run_best_value < self.global_best_value:
            self.global_best_value = run_best_value
            self.global_best_position = run_best_position.copy()
        
        convergence = []
        
        for iteration in range(self.max_iter):
            progress = iteration / self.max_iter
            #w = self.w_max - (self.w_max - self.w_min) * progress
            w = np.random.uniform(self.w_min, self.w_max)
            c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
            c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (p_best_positions[i] - positions[i]) +
                                 c2 * r2 * (self.global_best_position - positions[i]))
                
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], [b[0] for b in self.bounds], [b[1] for b in self.bounds])
                
                f_value = self.objective_function(*positions[i])
                
                if f_value < p_best_values[i]:
                    p_best_values[i] = f_value
                    p_best_positions[i] = positions[i].copy()
                    
                    if f_value < run_best_value:
                        run_best_value = f_value
                        run_best_position = positions[i].copy()
                        
                        if f_value < self.global_best_value:
                            self.global_best_value = f_value
                            self.global_best_position = positions[i].copy()
            
            convergence.append(run_best_value)
            print(f"Run {run_number + 1}, Iteration {iteration + 1}/{self.max_iter}, Best Value: {run_best_value:.6f}")
        
        return run_best_position, run_best_value, convergence

    def optimize(self):
        start_time = time.time()
        
        for run in range(self.n_restarts):
            print(f"\nStarting optimization run {run + 1}/{self.n_restarts}")
            run_best_position, run_best_value, run_convergence = self.optimize_single_run(run)
            self.all_runs_convergence.append(run_convergence)
            self.all_runs_best_positions.append(run_best_position)
            self.all_runs_best_values.append(run_best_value)
            print(f"Run {run + 1} complete - Best Value: {run_best_value:.6f}")
        
        best_run_idx = np.argmin(self.all_runs_best_values)
        self.convergence = self.all_runs_convergence[best_run_idx]
        self.computation_time = time.time() - start_time
        
        return self.global_best_position, self.global_best_value

    def plot_convergence(self):
        plt.figure(figsize=(12, 6))
        for i, conv in enumerate(self.all_runs_convergence):
            plt.plot(conv, label=f'Run {i+1} (Best: {self.all_runs_best_values[i]:.6f})')
        
        best_run_idx = np.argmin(self.all_runs_best_values)
        plt.plot(self.all_runs_convergence[best_run_idx], 'r-', linewidth=3, label=f'Best Run (Value: {self.global_best_value:.6f})')
        plt.xlabel("Iteration")
        plt.ylabel("Best Objective Value")
        plt.title("Convergence of PSO (All Runs)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_result(self):
        y1 = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
        y2 = np.linspace(self.bounds[1][0], self.bounds[1][1], 200)
        Y1, Y2 = np.meshgrid(y1, y2)
        Z = self.objective_function(Y1, Y2)
        
        plt.figure(figsize=(10, 6))
        plt.contourf(Y1, Y2, Z, levels=50, cmap="viridis")
        plt.colorbar(label="Objective Value")
        
        for i, pos in enumerate(self.all_runs_best_positions):
            plt.scatter(pos[0], pos[1], label=f'Run {i+1}', edgecolors='black')
        
        plt.scatter(self.global_best_position[0], self.global_best_position[1], color="red", s=100, marker='*', label="Global Best")
        plt.xlabel("y1")
        plt.ylabel("y2")
        plt.legend()
        plt.title("PSO Optimization Result")
        plt.show()

if __name__ == "__main__":
    def objective_function(y1, y2):
        return (1 - y1**2) * np.cos(5 * y1) + (12 - y2**2) * np.cos(4 / (np.sqrt(y2) + 0.3))
    
    bounds = [(0, 2), (0, 2)]
    pso = ImprovedParticleSwarmOptimizer(objective_function, bounds, num_particles=100, max_iter=200, n_restarts=5)
    best_position, best_value = pso.optimize()
    print(f"Best Position: {best_position}, Best Value: {best_value}")
    pso.plot_convergence()
    pso.plot_result()
