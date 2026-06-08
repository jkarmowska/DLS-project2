import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'visualisations'
OUTPUT_DIR.mkdir(exist_ok=True)

# List of files to process
files = [BASE_DIR / 'subtaskA_restored_metrics.json', BASE_DIR / 'subtaskB_restored_metrics.json']

for file in files:
    try:
        with file.open('r') as f:
            data = json.load(f)
            
        for seq in data.get('sequences', []):
            seq_id = seq['id']
            history = seq.get('history', [])
            
            # Extract models from initial scores
            models = list(seq['initial_scores'].keys())
            
            # Initialize lists for steps and scores starting at step 0
            steps = [0]
            scores = {m: [seq['initial_scores'][m]] for m in models}
            avg_scores = [np.mean([seq['initial_scores'][m] for m in models])]
            
            # Parse the optimization history
            for h in history:
                steps.append(h['step'])
                step_scores = []
                for m in models:
                    val = h['scores'][m]
                    scores[m].append(val)
                    step_scores.append(val)
                avg_scores.append(np.mean(step_scores))
                    
            # Create the visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot individual model scores with slight transparency
            for m in models:
                ax.plot(steps, scores[m], label=m, marker='o', markersize=3, alpha=0.6)
                
            # Plot the average score as a thicker black line
            ax.plot(steps, avg_scores, label='Average Score', color='black', linewidth=2.5, marker='s', markersize=4)
                
            # Formatting the plot
            ax.set_title(f'Score Optimization Process - {seq_id}')
            ax.set_xlabel('Mutation Step')
            ax.set_ylabel('Score')
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the plot
            filename = OUTPUT_DIR / f'{seq_id}_optimization.png'
            plt.savefig(filename)
            plt.close()
            
            print(f"Generated plot: {filename}")
            
    except FileNotFoundError:
        print(f"File {file} not found.")