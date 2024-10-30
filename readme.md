# Project Allocation Optimization Tool

## Overview

this tool is for optimizing project allocations when you're competing with others who have their own ranked preferences for projects.

using a mix of simulated annealing and the hungarian algorithm, we iterate over different possible rankings you could submit and find the best ranking that maximizes your expected utility — basically, the ranking that gives you the best shot at landing a project you want.

## How It Works
1. **Input CSVs**: you need two CSV files:
   - **Project Rankings CSV**: contains the ranked preferences of all the other students -- this is provided scraped as of 30/10/2024.
   - **Personal Rankings CSV**: your initial ranking of the projects -- this is provided by default as a as-scraped order. 

2. **Simulated Annealing**: this use simulated annealing to find an optimal ranking.

3. **Assignment Probability Calculation**: once we have an optimal ranking, this runs simulations to see how likely you are to be assigned each project for your own reference. this gives you a sense of what to expect, with some projects having higher assignment probabilities.

## Running the Tool

use the following command to run the script:

```bash
python project_allocation.py <project_rankings_path> <personal_ranking_path>
```

- `<project_rankings_path>`: path to the CSV file containing rankings from other students.
- `<personal_ranking_path>`: path to your personal ranking CSV file.

### Example Input
let's say you've got a file named `student_rankings.csv` and your own ranking file named `my_ranking.csv`. you'd run:

```bash
python project_allocation.py student_rankings.csv my_ranking.csv
```

### Output
- **Optimal Ranking**: it prints out your best possible ranking of projects that maximizes your expected chances of landing a top pick.
- **Optimal Expected Utility**: a numerical value representing the overall quality of your expected outcome, considering the competition.
- **Assignment Probabilities**: it shows the top projects along with their probabilities of assignment. only the top few are displayed, just to keep it clean.

## Dependencies
- `pandas`
- `numpy`
- `scipy`

make sure you install these via pip if you don't have them already:

```bash
pip install pandas numpy scipy
```

## Notes
- this uses multiprocessing to speed up the simulations, so you might want to run it on a machine with multiple cores for faster results.
- the annealing process can be adjusted with `num_iterations` and `num_simulations` — you can play with these to balance speed and accuracy. the default values should be good enough for most cases, runs within a few minutes on an M3.
