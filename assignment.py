import pandas as pd
import numpy as np
import random
import math
from scipy.optimize import linear_sum_assignment
import sys
from multiprocessing import Pool, cpu_count

def load_data(file_path):
    return pd.read_csv(file_path)

def load_personal_ranking(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values('ranking')  
    projects = df['project'].tolist()
    rankings = df['ranking'].tolist()
    return projects, rankings

def assign_utilities_from_ranking(rankings, projects):
    N = len(projects)
    return {project: N - rank for project, rank in zip(projects, rankings)}

def generate_other_students_rankings(args):
    df, num_students, projects, K = args
    frequency = {p: {k: 0 for k in range(1, K + 1)} for p in projects}
    for _, row in df.iterrows():
        for k in range(1, K + 1):
            project = row.get(f'Rank{k}', None)
            if pd.notna(project):
                frequency[project][k] += 1

    rank_prob = {}
    for k in range(1, K + 1):
        total = sum(frequency[p][k] for p in projects)
        rank_prob[k] = {p: frequency[p][k] / total if total > 0 else 1 / len(projects) for p in projects}

    simulated_rankings = []
    for _ in range(num_students):
        ranking = []
        available_projects = projects.copy()
        for k in range(1, K + 1):
            probs = [rank_prob[k][p] if p in available_projects else 0 for p in projects]
            total_prob = sum(probs)
            normalized_probs = [p / total_prob for p in probs]
            selected = random.choices(projects, weights=normalized_probs, k=1)[0]
            ranking.append(selected)
            available_projects.remove(selected)
        simulated_rankings.append(ranking)
    return simulated_rankings

def construct_cost_matrix(all_rankings, your_index, projects, K=6):
    M, N = len(all_rankings), len(projects)
    cost_matrix = np.full((M, N), K + 1)
    project_to_index = {p: i for i, p in enumerate(projects)}
    for i, ranking in enumerate(all_rankings):
        for rank, p in enumerate(ranking, start=1):
            j = project_to_index[p]
            cost_matrix[i, j] = rank
    return cost_matrix

def hungarian_algorithm_assignment(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {i: j for i, j in zip(row_ind, col_ind)}

def simulate_assignment(your_ranking, df, projects, K=6, num_simulations=10):
    num_other_students = len(df)
    args = [(df, num_other_students, projects, K) for _ in range(num_simulations)]
    with Pool(cpu_count()) as pool:
        other_rankings = pool.map(generate_other_students_rankings, args)

    assignment_counts = {p: 0 for p in projects}
    for ranking in other_rankings:
        all_rankings = ranking + [your_ranking]
        your_index = len(all_rankings) - 1
        cost_matrix = construct_cost_matrix(all_rankings, your_index, projects, K)
        assignments = hungarian_algorithm_assignment(cost_matrix)
        your_project_index = assignments.get(your_index, None)
        if your_project_index is not None:
            your_project = projects[your_project_index]
            assignment_counts[your_project] += 1
    probabilities = {p: count / num_simulations for p, count in assignment_counts.items()}
    return probabilities

def calculate_expected_utility(probabilities, utilities):
    return sum(utilities[p] * probabilities.get(p, 0) for p in utilities)

def mutate_ranking(ranking):
    new_ranking = ranking.copy()
    idx1, idx2 = random.sample(range(len(new_ranking)), 2)
    new_ranking[idx1], new_ranking[idx2] = new_ranking[idx2], new_ranking[idx1]
    return new_ranking

def simulated_annealing_optimize(initial_ranking, utilities, df, projects, K=6, initial_temp=1000, cooling_rate=0.95, num_iterations=50, num_simulations=10):
    current_ranking = initial_ranking.copy()
    current_E_U = calculate_expected_utility(simulate_assignment(current_ranking, df, projects, K, num_simulations), utilities)

    best_ranking = current_ranking.copy()
    best_E_U = current_E_U
    T = initial_temp

    for iteration in range(num_iterations):
        neighbor_ranking = mutate_ranking(current_ranking)
        neighbor_E_U = calculate_expected_utility(simulate_assignment(neighbor_ranking, df, projects, K, num_simulations), utilities)
        
        delta_E_U = neighbor_E_U - current_E_U
        if delta_E_U > 0 or math.exp(delta_E_U / T) > random.random():
            current_ranking = neighbor_ranking
            current_E_U = neighbor_E_U
            if current_E_U > best_E_U:
                best_ranking = current_ranking.copy()
                best_E_U = current_E_U
        T *= cooling_rate
        if T < 1e-3:
            break
    return best_ranking, best_E_U

def simulate_assignment_probability(best_ranking, df, projects, K=6, num_simulations=100):
    num_other_students = len(df)
    args = [(df, num_other_students, projects, K) for _ in range(num_simulations)]
    with Pool(cpu_count()) as pool:
        other_rankings = pool.map(generate_other_students_rankings, args)

    assignment_counts = {p: 0 for p in projects}
    for ranking in other_rankings:
        all_rankings = ranking + [best_ranking]
        your_index = len(all_rankings) - 1
        cost_matrix = construct_cost_matrix(all_rankings, your_index, projects, K)
        assignments = hungarian_algorithm_assignment(cost_matrix)
        your_project_index = assignments.get(your_index, None)
        if your_project_index is not None:
            your_project = projects[your_project_index]
            assignment_counts[your_project] += 1
    probabilities = {p: count / num_simulations for p, count in assignment_counts.items()}
    return probabilities

def main(project_rankings_path, personal_ranking_path):
    df = load_data(project_rankings_path)
    projects, rankings = load_personal_ranking(personal_ranking_path)
    utilities = assign_utilities_from_ranking(rankings, projects)
    best_ranking, best_expected_utility = simulated_annealing_optimize(
        initial_ranking=projects, 
        utilities=utilities,
        df=df, 
        projects=projects, 
        K=6, 
        initial_temp=1000, 
        cooling_rate=0.95, 
        num_iterations=100, 
        num_simulations=100
    )
    final_probabilities = simulate_assignment_probability(best_ranking, df, projects, K=6, num_simulations=50)
    prob_df = pd.DataFrame(list(final_probabilities.items()), columns=['Project', 'Assignment Probability']).sort_values(
        by='Assignment Probability', ascending=False)
    print("Optimal Ranking:", best_ranking)
    print("Optimal Expected Utility:", best_expected_utility)
    print("Approximate Assignment Probabilities Under This Ranking")
    print(prob_df)

if __name__ == "__main__":
    project_rankings_path = sys.argv[1]
    personal_ranking_path = sys.argv[2]
    main(project_rankings_path, personal_ranking_path)
