import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os

# Define the Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# PSO Algorithm with stopping criteria and convergence plot customization
def pso_rosenbrock(num_particles=60, max_iter=50, dim=2, w=0.5, c1=1.5, c2=2, patience=10, tolerance=1e-5):
    positions = np.random.uniform(-2, 2, (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    p_best_positions = positions.copy()
    p_best_values = np.array([rosenbrock(p) for p in positions])
    g_best_position = p_best_positions[np.argmin(p_best_values)]
    g_best_value = np.min(p_best_values)

    convergence = []
    no_improvement_counter = 0
    start_time = time.time()
    process = psutil.Process(os.getpid())

    for iteration in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (p_best_positions[i] - positions[i])
                + c2 * r2 * (g_best_position - positions[i])
            )
            positions[i] += velocities[i]
            f_value = rosenbrock(positions[i])
            if f_value < p_best_values[i]:
                p_best_values[i] = f_value
                p_best_positions[i] = positions[i]

        new_g_best_value = np.min(p_best_values)
        if abs(g_best_value - new_g_best_value) < tolerance:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        g_best_position = p_best_positions[np.argmin(p_best_values)]
        g_best_value = new_g_best_value
        convergence.append(g_best_value)

        if no_improvement_counter >= patience:
            print(f"Stopping early at iteration {iteration + 1} due to lack of improvement.")
            break

        print(f"Iteration {iteration + 1}/{max_iter}, Best Value: {g_best_value:.6f}")

    end_time = time.time()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # in MB
    computation_time = end_time - start_time

    return g_best_position, g_best_value, convergence, computation_time, memory_usage

# Function to plot convergence analysis
def plot_convergence_analysis():
    particle_sizes = [30, 60, 100]
    w_values = [0.3, 0.5, 0.7]
    max_iter = 50

    plt.figure(figsize=(10, 8))

    # Keeping w constant and varying particle size
    for size in particle_sizes:
        _, _, convergence, _, _ = pso_rosenbrock(num_particles=size, max_iter=max_iter, w=0.5)
        plt.plot(convergence, label=f"Particles={size}, w=0.5")

    plt.title("Convergence: Constant w, Varying Particle Size")
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Keeping particle size constant and varying w
    plt.figure(figsize=(10, 8))
    for w in w_values:
        _, _, convergence, _, _ = pso_rosenbrock(num_particles=60, max_iter=max_iter, w=w)
        plt.plot(convergence, label=f"Particles=60, w={w}")

    plt.title("Convergence: Constant Particle Size, Varying w")
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def sensitivity_analysis(best_position, perturbation=0.01, samples=5):
    print("\nPerforming sensitivity analysis...")
    sensitivities = []
    
    for i in range(samples):
        perturbed_position = best_position + np.random.uniform(-perturbation, perturbation, best_position.shape)
        perturbed_value = rosenbrock(perturbed_position)
        sensitivities.append(perturbed_value)
        print(f"Perturbed Position: {perturbed_position}, Value: {perturbed_value:.6f}")
    
    avg_perturbed_value = np.mean(sensitivities)
    print(f"Average Perturbed Value: {avg_perturbed_value:.6f}")
    return sensitivities

# Main execution
if __name__ == "__main__":
    # Run and plot convergence analysis
    plot_convergence_analysis()

    # Run PSO with default parameters and stopping criteria
    best_position, best_value, convergence, computation_time, memory_usage = pso_rosenbrock()

    print("\nOptimization Results:")
    print(f"Best Position: {best_position}")
    print(f"Best Value: {best_value}")
    print(f"Computation Time: {computation_time:.4f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Plot convergence
    plt.figure(figsize=(8, 6))
    plt.plot(convergence, label='Best Objective Value', color='red')
    plt.title("Convergence of PSO with Stopping Criteria")
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    sensitivity_analysis(best_position)


