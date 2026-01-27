from src.genetic import GeneticAlgorithm
from src.rl_agent import QLearningAgent
from src.trainer import evaluate_config
import matplotlib.pyplot as plt
import time
import numpy as np

POP_SIZE = 20        
GENERATIONS = 10     
RL_EPISODES = POP_SIZE * GENERATIONS # = 200, to have the same budget

def run_genetic_vs_random():
    print(f"\n[MODE 1] Detailed Analysis: Genetic vs Random...")
    start = time.time()
    
    # GENETIC ALGORITHM
    ga = GeneticAlgorithm(pop_size=POP_SIZE, mutation_rate=0.25)
    pop = ga.init_population()
    
    hist_ga_best = []
    hist_ga_avg = []
    global_ga_best = float('inf')
    
    print("  > Run GA...")
    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen+1}/{GENERATIONS} ---")
        scores = []
        for i, ind in enumerate(pop):
            loss = evaluate_config(ind)
            scores.append(loss)
            print(f"    > Ind {i+1:02d}: Loss={loss:.4f} | LR={ind['lr']:.5f}")
        
        # Stats GA
        gen_best = min(scores)
        gen_avg = sum(scores) / len(scores)
        
        if gen_best < global_ga_best: global_ga_best = gen_best
        
        hist_ga_best.append(global_ga_best)
        hist_ga_avg.append(gen_avg)
        
        print(f"Stats Gen {gen+1}: Global Record={global_ga_best:.4f} | Avg={gen_avg:.4f}")
        
        if gen < GENERATIONS - 1:
            pop = ga.evolve(pop, scores)

    # RANDOM SEARCH (By Batch)
    print("\n  > Run Random...")
    ga_rnd = GeneticAlgorithm(pop_size=1)
    
    hist_rnd_best = []
    hist_rnd_avg = []
    global_rnd_best = float('inf')
    
    for gen in range(GENERATIONS):
        print(f"\n--- Random Batch {gen+1}/{GENERATIONS} ---")
        batch_scores = []
        for i in range(POP_SIZE):
            cfg = ga_rnd.init_population()[0]
            loss = evaluate_config(cfg)
            batch_scores.append(loss)
            print(f"    > Random Trial {i+1:02d}: Loss={loss:.4f}")
            
        # Stats Random
        batch_min = min(batch_scores)
        batch_avg = sum(batch_scores) / len(batch_scores)
        
        if batch_min < global_rnd_best: global_rnd_best = batch_min
        
        hist_rnd_best.append(global_rnd_best)
        hist_rnd_avg.append(batch_avg)
        
        print(f"Stats Batch {gen+1}: Record={global_rnd_best:.4f} | Batch Avg={batch_avg:.4f}")

    print(f"Finished in {time.time()-start:.1f}s")
    
    plt.figure(figsize=(12, 7))
    x = range(1, GENERATIONS+1)
    
    # GA
    plt.plot(x, hist_ga_best, 'b-o', linewidth=3, label='GA: Record (Best So Far)')
    plt.plot(x, hist_ga_avg, 'b--', alpha=0.5, label='GA: Pop Average (Learning)')
    
    # Random
    plt.plot(x, hist_rnd_best, 'r-x', linewidth=2, label='Random: Record (Luck)')
    plt.plot(x, hist_rnd_avg, 'r:', alpha=0.5, label='Random: Average (Stagnation)')
    
    plt.title("Detailed Analysis: Intelligence (GA) vs Random")
    plt.xlabel("Generations")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("result_mode1_ga_vs_random.png")
    plt.show()

def run_rl_detailed():
    print(f"\n[MODE 2] Detailed Analysis: Reinforcement Learning (Internal params)...")
    start = time.time()
    
    # Initialize my agent
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1)
    
    # Start the training loop
    best_config, best_loss, rewards_history = agent.train(episodes=RL_EPISODES)
    
    # Process data for visualization
    # The agent returns rewards, but I want to plot Loss (easier to read)
    hist_rl_raw = [10.0 / (r + 1e-8) for r in rewards_history]
    
    # Create a "Best So Far" curve to make the graph look cleaner
    hist_rl_best = []
    current_best = float('inf')
    for loss in hist_rl_raw:
        if loss < current_best:
            current_best = loss
        hist_rl_best.append(current_best)

    print(f"\nFinished in {time.time()-start:.1f}s")
    print(f"Best Config Found: {best_config}")
    print(f"Best Loss: {best_loss:.4f}")

    plt.figure(figsize=(12, 7))
    
    # Light green dots: Show every single attempt (Exploration noise)
    plt.plot(hist_rl_raw, 'o', color='green', alpha=0.3, markersize=4, label='RL: Raw Score')
    
    # Solid green line: Shows the convergence of the best result
    plt.plot(hist_rl_best, 'g-', linewidth=3, label='RL: Best So Far')
    
    plt.title("Reinforcement Learning Performance")
    plt.xlabel("Episodes")
    plt.ylabel("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("result_mode2_rl_details.png")
    plt.show()

def main():
    while True:
        print("\n" + "="*40)
        print("   AI PROJECT MENU - DATA SCIENCE")
        print("="*40)
        print("1. Analysis GA vs Random (Best & Avg)")
        print("2. Analysis RL (Best & Raw)")
        print("3. Quit")
        
        c = input("\nChoice : ")
        if c=='1': run_genetic_vs_random()
        elif c=='2': run_rl_detailed()
        elif c=='3': break
        else: print("?")

if __name__ == "__main__":
    main()