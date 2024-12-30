# Python Optimization Libraries

Python offers numerous powerful libraries for optimization tasks. Depending on your needs (e.g., linear programming, nonlinear optimization, global optimization), here are some commonly used libraries:

---

## **General Optimization Tools**

### 1. **SciPy**
- **Module**: `scipy.optimize`
- **Features**: Supports linear programming, nonlinear programming, minimization, multi-dimensional function optimization, equation solving, and more.
- **Installation**: `pip install scipy`
- **Example**:

```python
from scipy.optimize import minimize

# Define objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Initial values
x0 = [1, 1]

# Optimization
result = minimize(objective, x0)
print(result)
```

### 2. **CVXPY**
- **Features**: Solves convex optimization problems, supporting linear programming, quadratic programming, etc.
- **Installation**: `pip install cvxpy`
- **Example**:

```python
import cvxpy as cp

# Define variables
x = cp.Variable()

# Define problem
objective = cp.Minimize((x - 2)**2)
constraints = [x >= 0]

# Solve
prob = cp.Problem(objective, constraints)
prob.solve()
print("Optimal value:", prob.value)
print("Optimal variable:", x.value)
```

### 3. **CVXOPT**
- **Features**: A low-level library for convex optimization. It provides efficient solvers for quadratic programming (QP), linear programming (LP), and conic programming.
- **Installation**: `pip install cvxopt`
- **Example**:

```python
from cvxopt import matrix, solvers

# Define quadratic programming problem
Q = matrix([[2.0, 0.0], [0.0, 2.0]])  # Quadratic term
p = matrix([-4.0, -6.0])  # Linear term
G = matrix([[-1.0, 0.0], [0.0, -1.0]])  # Inequality constraint matrix
h = matrix([0.0, 0.0])  # Inequality constraint bounds

# Solve
solution = solvers.qp(Q, p, G, h)
print(solution['x'])  # Optimal solution
```

---

## **Domain-Specific Tools**

### 3. **Pyomo**
- **Features**: Used for modeling mathematical programming problems, supporting linear, nonlinear, and integer programming.
- **Installation**: `pip install pyomo`
- **Example**:

```python
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(bounds=(0, None))
model.obj = Objective(expr=model.x**2 - 4*model.x + 4, sense=minimize)

SolverFactory('glpk').solve(model)
print("Optimal x:", model.x.value)
```

### 4. **PuLP**
- **Features**: Focused on linear programming (LP).
- **Installation**: `pip install pulp`
- **Example**:

```python
from pulp import LpMaximize, LpProblem, LpVariable

# Define problem
prob = LpProblem("Maximize_Profit", LpMaximize)
x = LpVariable("x", lowBound=0)
y = LpVariable("y", lowBound=0)

# Objective function
prob += 4 * x + 3 * y

# Constraints
prob += 2 * x + y <= 8
prob += x + 2 * y <= 6

# Solve
prob.solve()
print("Optimal Solution:", x.value(), y.value())
```

---

## **Global Optimization Tools**

### 5. **DEAP**
- **Features**: Genetic algorithms and evolutionary computation.
- **Installation**: `pip install deap`
- **Example**:

```python
from deap import base, creator, tools, algorithms
import random

# Fitness function
def eval_func(individual):
    return sum(individual),

# Configure genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run genetic algorithm
population = toolbox.population(n=50)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)
best = tools.selBest(population, 1)[0]
print("Best Individual:", best)
```

### 6. **Optuna**
- **Features**: For hyperparameter tuning and black-box optimization.
- **Installation**: `pip install optuna`
- **Example**:

```python
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2)**2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print("Best Parameter:", study.best_params)
```

---
