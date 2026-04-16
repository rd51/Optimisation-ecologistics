"""
================================================================================
MICROGRID ENERGY DISPATCH & BATTERY STORAGE OPTIMIZATION
Interactive Streamlit Dashboard with Complete Mathematical Model Formulation
================================================================================

PROBLEM DESCRIPTION:
    A microgrid operator must optimize hourly energy dispatch over 24 hours,
    balancing:
      - Real-time electricity load demand
      - Solar photovoltaic generation (weather-dependent)
      - Grid energy purchases (subject to Time-of-Use pricing)
      - Battery energy storage system (BESS) charge/discharge schedules
    
    Goal: MINIMIZE TOTAL ENERGY COST while satisfying all constraints.
    
    Real-world applications:
      - Commercial buildings with on-site solar + battery storage
      - Isolated communities/islands with renewable energy and backup storage
      - Peak shaving to reduce demand charges
      - Load shifting to exploit TOU pricing arbitrage

MODEL TYPE: Linear Programming (LP)
SOLVER: PuLP with CBC backend
PLANNING HORIZON: 24 hours (hourly intervals)
COMPLEXITY: 72 variables, ~100 constraints

MATHEMATICAL FOUNDATION:
    Decision: g_t, c_t, d_t, b_t for t=0..23
    Minimise: Z = Σ price_t × g_t
    Subject to: Energy balance, SOC dynamics, capacity/rate limits
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import streamlit as st
st.set_page_config(layout="wide", page_title="Microgrid Energy Optimization")

import pulp
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SETUP & SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

HOURS = 24
TIME_STEPS = list(range(HOURS))
BASE_HOURS = [datetime(2024, 1, 15, hour=h) for h in range(HOURS)]

# Battery system parameters
BATTERY_CAPACITY_KWH = 100.0
MAX_CHARGE_POWER_KW = 25.0
MAX_DISCHARGE_POWER_KW = 25.0
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
INITIAL_SOC_PCT = 0.20
FINAL_SOC_PCT = 0.50

# Solar capacity (installed)
SOLAR_CAPACITY_KWP = 30.0

# Multi-objective dispatch parameters (cost and emissions per kWh)
SOLAR_COST_PER_KWH = 0.01
BATTERY_COST_PER_KWH = 0.03
DIESEL_COST_PER_KWH = 0.35

SOLAR_EMISSIONS_KG_PER_KWH = 0.0
BATTERY_EMISSIONS_KG_PER_KWH = 0.05
DIESEL_EMISSIONS_KG_PER_KWH = 0.75

MAX_DIESEL_POWER_KW = 120.0

def generate_solar_profile():
    """Realistic solar generation profile (bell curve peaking at noon)."""
    solar = np.zeros(HOURS)
    for h in range(6, 19):
        normalized = np.sin(np.pi * (h - 6) / (18 - 6)) ** 2
        solar[h] = SOLAR_CAPACITY_KWP * normalized * (1 + 0.05 * np.random.randn())
    return np.maximum(solar, 0)

def generate_load_demand():
    """Typical commercial building demand profile."""
    load = np.zeros(HOURS)
    load[:6] = 20.0  # Night baseload
    load[23] = 20.0  # Late evening
    
    for h in range(6, 9):
        load[h] = 20 + (h - 6) * 8  # Morning ramp
    
    load[9:17] = 44 + 10 * np.sin(np.pi * (np.arange(8)) / 8)  # Daytime
    
    for h in range(17, 21):
        peak_factor = np.sin(np.pi * (h - 17) / 4)
        load[h] = 44 + 15 * peak_factor  # Evening peak
    
    for h in range(21, 24):
        load[h] = 20 + (24 - h) * 3  # Evening ramp down
    
    load += 0.5 * np.random.randn(HOURS)
    return np.maximum(load, 15)

def generate_tou_pricing():
    """Time-of-Use pricing (realistic UK tariff)."""
    price = np.zeros(HOURS)
    for h in range(HOURS):
        if h < 7:
            price[h] = 0.15   # Night: cheap
        elif h < 16:
            price[h] = 0.20   # Off-peak day
        elif h < 21:
            peak_factor = (h - 16) / 5
            price[h] = 0.20 + peak_factor * 0.15  # Evening peak (ramps 0.20→0.35)
        else:
            price[h] = 0.25   # Late evening
    return price

# Generate data
SOLAR_GENERATION = generate_solar_profile()
LOAD_DEMAND = generate_load_demand()
GRID_PRICE = generate_tou_pricing()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BASELINE (NO BATTERY)
# ─────────────────────────────────────────────────────────────────────────────

def compute_baseline_cost(load, solar, price):
    """Baseline: All demand not met by solar from grid (no optimization)."""
    grid_purchase = np.maximum(load - solar, 0)
    cost = np.sum(grid_purchase * price)
    return cost, grid_purchase

BASELINE_COST, BASELINE_GRID = compute_baseline_cost(LOAD_DEMAND, SOLAR_GENERATION, GRID_PRICE)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: OPTIMIZATION MODEL WITH EQUATION MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def build_optimize_microgrid_model(load, solar, price, battery_cap, charge_power,
                                    discharge_power, charge_eff=0.95, 
                                    discharge_eff=0.95, init_soc=0.2, final_soc=0.5):
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                   MATHEMATICAL MODEL FORMULATION                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    DECISION VARIABLES:
        g_t ∈ ℝ⁺    Grid energy purchase at hour t (kW)
        c_t ∈ ℝ⁺    Battery charging power at hour t (kW)
        d_t ∈ ℝ⁺    Battery discharging power at hour t (kW)
        b_t ∈ ℝ⁺    Battery state of charge at hour t (kWh)
    
    OBJECTIVE FUNCTION:
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃  min Z = Σ_t [ price_t × g_t ]           ┃
        ┃                                           ┃
        ┃  Minimize total daily energy cost         ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    
    CONSTRAINTS:
    
    C1: ENERGY BALANCE (Power conservation)
        load_t = g_t + d_t + solar_t - c_t    ∀t ∈ {0..23}
        ↓
        Interpretation: Total power in = Total power out
        Left: Demand to satisfy
        Right: Grid + Battery discharge + Solar - Battery charge
    
    C2: BATTERY STATE-OF-CHARGE DYNAMICS (Battery physics)
        b_{t+1} = b_t + η_c×c_t - (1/η_d)×d_t    ∀t ∈ {0..22}
        ↓
        Interpretation: Energy accumulation with losses
        Next hour's battery = This hour's + Energy charged (w/ losses) - Energy extracted (w/ losses)
        η_c, η_d account for round-trip inefficiency
    
    C3: BATTERY CAPACITY LIMIT (Physical constraint)
        0 ≤ b_t ≤ B_max    ∀t
        ↓
        Current stored energy cannot exceed tank capacity
    
    C4: CHARGING POWER LIMIT (Battery power rating)
        0 ≤ c_t ≤ P_c,max    ∀t
        ↓
        Battery charger cannot supply more than rated power
    
    C5: DISCHARGING POWER LIMIT (Battery power rating)
        0 ≤ d_t ≤ P_d,max    ∀t
        ↓
        Battery inverter cannot deliver more than rated power
    
    C6: INITIAL STATE OF CHARGE (Known starting condition)
        b_0 = init_soc × B_max
        ↓
        Battery begins day with specified charge level
    
    C7: TERMINAL STATE OF CHARGE (Operational requirement)
        b_23 ≥ final_soc × B_max
        ↓
        Battery must maintain reserve at day end (prevents over-discharge)
    
    C8: NON-NEGATIVITY (Physical reality)
        g_t ≥ 0, c_t ≥ 0, d_t ≥ 0    ∀t
        ↓
        Cannot have negative power flows
    """
    
    prob = pulp.LpProblem("Microgrid_Energy_Dispatch", pulp.LpMinimize)
    
    # ──────────────────────────────────────────────────────────────────────────
    # DECISION VARIABLES
    # ──────────────────────────────────────────────────────────────────────────
    
    # Grid purchase (g_t)
    grid = {t: pulp.LpVariable(f"grid_{t}", lowBound=0, cat='Continuous') 
            for t in TIME_STEPS}
    
    # Battery charge (c_t)
    charge = {t: pulp.LpVariable(f"charge_{t}", lowBound=0, cat='Continuous') 
              for t in TIME_STEPS}
    
    # Battery discharge (d_t)
    discharge = {t: pulp.LpVariable(f"discharge_{t}", lowBound=0, cat='Continuous') 
                 for t in TIME_STEPS}
    
    # Battery SOC (b_t)
    soc = {t: pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=battery_cap, cat='Continuous') 
           for t in TIME_STEPS}
    
    # ──────────────────────────────────────────────────────────────────────────
    # OBJECTIVE FUNCTION
    # ──────────────────────────────────────────────────────────────────────────
    
    # EQUATION: Z = Σ_t [ price_t × g_t ]
    prob += pulp.lpSum(price[t] * grid[t] for t in TIME_STEPS), "Total_Energy_Cost"
    
    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRAINTS
    # ──────────────────────────────────────────────────────────────────────────
    
    for t in TIME_STEPS:
        
        # C1: ENERGY BALANCE
        # EQUATION: load_t = g_t + d_t + solar_t - c_t
        prob += (
            grid[t] + discharge[t] + solar[t] == load[t] + charge[t],
            f"energy_balance_{t}"
        )
        
        # C2: SOC TRANSITION
        # EQUATION: b_{t+1} = b_t + η_c×c_t - (1/η_d)×d_t
        if t < HOURS - 1:
            prob += (
                soc[t+1] == soc[t] + charge_eff * charge[t] - (1 / discharge_eff) * discharge[t],
                f"soc_transition_{t}"
            )
        
        # C4: CHARGE RATE LIMIT
        # EQUATION: c_t ≤ P_c,max
        prob += (charge[t] <= charge_power, f"charge_limit_{t}")
        
        # C5: DISCHARGE RATE LIMIT
        # EQUATION: d_t ≤ P_d,max
        prob += (discharge[t] <= discharge_power, f"discharge_limit_{t}")
    
    # C6: INITIAL SOC
    # EQUATION: b_0 = init_soc × B_max
    prob += (soc[0] == init_soc * battery_cap, "initial_soc")
    
    # C7: FINAL SOC
    # EQUATION: b_23 ≥ final_soc × B_max
    prob += (soc[HOURS - 1] >= final_soc * battery_cap, "final_soc")
    
    # ──────────────────────────────────────────────────────────────────────────
    # SOLVE
    # ──────────────────────────────────────────────────────────────────────────
    
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)
    
    # ──────────────────────────────────────────────────────────────────────────
    # EXTRACT RESULTS
    # ──────────────────────────────────────────────────────────────────────────
    
    results = {
        'status': pulp.LpStatus[prob.status],
        'objective': pulp.value(prob.objective),
        'grid': {t: pulp.value(grid[t]) for t in TIME_STEPS},
        'charge': {t: pulp.value(charge[t]) for t in TIME_STEPS},
        'discharge': {t: pulp.value(discharge[t]) for t in TIME_STEPS},
        'soc': {t: pulp.value(soc[t]) for t in TIME_STEPS},
    }
    
    return results, prob


def solve_multiobjective_dispatch(load, solar, battery_cap, charge_power, discharge_power,
                                  init_soc, final_soc, alpha, diesel_power_max,
                                  cost_factors, emission_factors):
    """Solve weighted-sum multi-objective dispatch (cost vs emissions)."""
    prob = pulp.LpProblem("Multiobjective_Dispatch", pulp.LpMinimize)

    solar_use = {t: pulp.LpVariable(f"solar_use_{t}", lowBound=0, upBound=solar[t], cat='Continuous')
                 for t in TIME_STEPS}
    diesel = {t: pulp.LpVariable(f"diesel_{t}", lowBound=0, upBound=diesel_power_max, cat='Continuous')
              for t in TIME_STEPS}
    charge = {t: pulp.LpVariable(f"m_charge_{t}", lowBound=0, cat='Continuous')
              for t in TIME_STEPS}
    discharge = {t: pulp.LpVariable(f"m_discharge_{t}", lowBound=0, cat='Continuous')
                 for t in TIME_STEPS}
    soc = {t: pulp.LpVariable(f"m_soc_{t}", lowBound=0, upBound=battery_cap, cat='Continuous')
           for t in TIME_STEPS}

    cost = pulp.lpSum(
        cost_factors["solar"] * solar_use[t]
        + cost_factors["battery"] * discharge[t]
        + cost_factors["diesel"] * diesel[t]
        for t in TIME_STEPS
    )
    emissions = pulp.lpSum(
        emission_factors["solar"] * solar_use[t]
        + emission_factors["battery"] * discharge[t]
        + emission_factors["diesel"] * diesel[t]
        for t in TIME_STEPS
    )

    prob += alpha * cost + (1 - alpha) * emissions, "Weighted_Objective"

    for t in TIME_STEPS:
        prob += (
            solar_use[t] + diesel[t] + discharge[t] == load[t] + charge[t],
            f"mo_energy_balance_{t}"
        )

        if t < HOURS - 1:
            prob += (
                soc[t+1] == soc[t] + CHARGE_EFFICIENCY * charge[t]
                - (1 / DISCHARGE_EFFICIENCY) * discharge[t],
                f"mo_soc_transition_{t}"
            )

        prob += (charge[t] <= charge_power, f"mo_charge_limit_{t}")
        prob += (discharge[t] <= discharge_power, f"mo_discharge_limit_{t}")

    prob += (soc[0] == init_soc * battery_cap, "mo_initial_soc")
    prob += (soc[HOURS - 1] >= final_soc * battery_cap, "mo_final_soc")

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    results = {
        "status": pulp.LpStatus[prob.status],
        "cost": pulp.value(cost),
        "emissions": pulp.value(emissions),
        "solar_use": {t: pulp.value(solar_use[t]) for t in TIME_STEPS},
        "diesel": {t: pulp.value(diesel[t]) for t in TIME_STEPS},
        "discharge": {t: pulp.value(discharge[t]) for t in TIME_STEPS},
    }

    return results


def simulate_dispatch(load, solar, battery_cap, charge_power, discharge_power,
                      init_soc, final_soc, diesel_power_max, discharge_schedule=None):
    """Simulate dispatch with optional battery discharge schedule."""
    soc = init_soc * battery_cap
    solar_use = np.zeros(HOURS)
    battery_discharge = np.zeros(HOURS)
    diesel_use = np.zeros(HOURS)
    unmet = np.zeros(HOURS)

    for t in TIME_STEPS:
        demand = load[t]
        solar_available = solar[t]

        solar_to_load = min(solar_available, demand)
        solar_use[t] = solar_to_load
        demand -= solar_to_load

        # Battery discharge decision
        if discharge_schedule is None:
            discharge_target = min(demand, discharge_power)
        else:
            discharge_target = min(discharge_schedule[t], discharge_power, demand)

        max_discharge_by_soc = soc * DISCHARGE_EFFICIENCY
        discharge_actual = min(discharge_target, max_discharge_by_soc)
        battery_discharge[t] = discharge_actual
        soc -= discharge_actual / DISCHARGE_EFFICIENCY
        demand -= discharge_actual

        # Diesel meets remaining demand
        diesel_actual = min(demand, diesel_power_max)
        diesel_use[t] = diesel_actual
        demand -= diesel_actual

        if demand > 0:
            unmet[t] = demand

        # Charge from leftover solar
        solar_surplus = solar_available - solar_to_load
        if solar_surplus > 0:
            charge_limit = min(charge_power, solar_surplus)
            capacity_remaining = battery_cap - soc
            max_charge_by_soc = capacity_remaining / CHARGE_EFFICIENCY
            charge_actual = min(charge_limit, max_charge_by_soc)
            soc += charge_actual * CHARGE_EFFICIENCY

    reliability = 100.0 * (1.0 - np.mean(unmet > 1e-6))

    return {
        "solar_use": solar_use,
        "battery_discharge": battery_discharge,
        "diesel_use": diesel_use,
        "unmet": unmet,
        "reliability": reliability
    }


def evaluate_dispatch_costs(dispatch, cost_factors, emission_factors):
    """Compute total cost and emissions for a dispatch outcome."""
    cost = (
        cost_factors["solar"] * np.sum(dispatch["solar_use"])
        + cost_factors["battery"] * np.sum(dispatch["battery_discharge"])
        + cost_factors["diesel"] * np.sum(dispatch["diesel_use"])
    )
    emissions = (
        emission_factors["solar"] * np.sum(dispatch["solar_use"])
        + emission_factors["battery"] * np.sum(dispatch["battery_discharge"])
        + emission_factors["diesel"] * np.sum(dispatch["diesel_use"])
    )
    return cost, emissions


def run_ga_dispatch(load, solar, battery_cap, charge_power, discharge_power,
                    init_soc, final_soc, diesel_power_max, cost_factors,
                    emission_factors, population_size=50, generations=100):
    """Simple GA for battery discharge schedule optimization."""
    rng = np.random.default_rng(42)
    gene_len = HOURS

    def init_individual():
        return rng.uniform(0, discharge_power, size=gene_len)

    def fitness(individual):
        dispatch = simulate_dispatch(
            load, solar, battery_cap, charge_power, discharge_power,
            init_soc, final_soc, diesel_power_max, discharge_schedule=individual
        )
        cost, emissions = evaluate_dispatch_costs(dispatch, cost_factors, emission_factors)
        unmet = np.sum(dispatch["unmet"])
        penalty = 1000.0 * unmet
        return cost + penalty

    population = [init_individual() for _ in range(population_size)]
    best_history = []
    best_individual = None
    best_fit = float("inf")

    for _ in range(generations):
        fitness_scores = np.array([fitness(ind) for ind in population])
        best_idx = int(np.argmin(fitness_scores))
        if fitness_scores[best_idx] < best_fit:
            best_fit = float(fitness_scores[best_idx])
            best_individual = population[best_idx].copy()
        best_history.append(best_fit)

        # Selection (tournament)
        selected = []
        for _ in range(population_size):
            i, j = rng.integers(0, population_size, size=2)
            winner = population[i] if fitness_scores[i] < fitness_scores[j] else population[j]
            selected.append(winner.copy())

        # Crossover
        next_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % population_size]
            if rng.random() < 0.7:
                point = rng.integers(1, gene_len - 1)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            next_population.extend([child1, child2])

        # Mutation
        for idx in range(len(next_population)):
            if rng.random() < 0.2:
                mutate_idx = rng.integers(0, gene_len)
                noise = rng.normal(0, discharge_power * 0.1)
                next_population[idx][mutate_idx] = np.clip(
                    next_population[idx][mutate_idx] + noise, 0, discharge_power
                )

        population = next_population[:population_size]

    best_dispatch = simulate_dispatch(
        load, solar, battery_cap, charge_power, discharge_power,
        init_soc, final_soc, diesel_power_max, discharge_schedule=best_individual
    )
    best_cost, best_emissions = evaluate_dispatch_costs(best_dispatch, cost_factors, emission_factors)

    return best_dispatch, best_cost, best_emissions, best_history


def build_grid_topology_plot(flows, metadata):
    """Build a Plotly figure showing power flow topology."""
    graph = nx.DiGraph()

    sources = ["Solar Panel", "Battery Storage", "Diesel Generator", "Grid Import"]
    sink = "Load Demand"

    for source in sources:
        graph.add_edge(source, sink, weight=flows.get(source, 0.0))

    pos = nx.spring_layout(graph, seed=7)

    edge_traces = []
    max_flow = max(flows.values()) if flows else 1.0
    max_flow = max(max_flow, 1e-6)

    edge_colors = {
        "Solar Panel": "#ffcc00",
        "Battery Storage": "#2ca02c",
        "Diesel Generator": "#d62728",
        "Grid Import": "#1f77b4",
    }

    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        flow = data.get("weight", 0.0)
        width = 1.5 + 6.0 * (flow / max_flow)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=edge_colors.get(source, "#888")),
                hoverinfo="text",
                text=f"{source} → {target}<br>Flow: {flow:.2f} kW",
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_text = []
    node_hover = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        meta = metadata.get(node, {})
        hover = (
            f"{node}<br>Capacity: {meta.get('capacity', 'N/A')} kW"
            f"<br>Output: {meta.get('output', 0.0):.2f} kW"
            f"<br>Cost: {meta.get('cost', 'N/A')} £/kWh"
        )
        node_hover.append(hover)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(size=22, color="#f2f2f2", line=dict(width=1.5, color="#333"))
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Optimal Power Flow — Current Dispatch",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def solve_horizon_lp(load, solar, price, battery_cap, charge_power, discharge_power,
                     init_soc, final_soc_required):
    """Solve a short-horizon LP and return the first-hour decisions."""
    horizon = len(load)
    steps = list(range(horizon))
    prob = pulp.LpProblem("Rolling_Horizon", pulp.LpMinimize)

    grid = {t: pulp.LpVariable(f"rh_grid_{t}", lowBound=0, cat='Continuous') for t in steps}
    charge = {t: pulp.LpVariable(f"rh_charge_{t}", lowBound=0, cat='Continuous') for t in steps}
    discharge = {t: pulp.LpVariable(f"rh_discharge_{t}", lowBound=0, cat='Continuous') for t in steps}
    soc = {t: pulp.LpVariable(f"rh_soc_{t}", lowBound=0, upBound=battery_cap, cat='Continuous') for t in steps}

    prob += pulp.lpSum(price[t] * grid[t] for t in steps), "Rolling_Cost"

    for t in steps:
        prob += (
            grid[t] + discharge[t] + solar[t] == load[t] + charge[t],
            f"rh_energy_balance_{t}"
        )
        if t < horizon - 1:
            prob += (
                soc[t+1] == soc[t] + CHARGE_EFFICIENCY * charge[t]
                - (1 / DISCHARGE_EFFICIENCY) * discharge[t],
                f"rh_soc_transition_{t}"
            )
        prob += (charge[t] <= charge_power, f"rh_charge_limit_{t}")
        prob += (discharge[t] <= discharge_power, f"rh_discharge_limit_{t}")

    prob += (soc[0] == init_soc * battery_cap, "rh_initial_soc")
    if final_soc_required:
        prob += (soc[horizon - 1] >= final_soc_required * battery_cap, "rh_final_soc")

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    grid0 = pulp.value(grid[0])
    charge0 = pulp.value(charge[0])
    discharge0 = pulp.value(discharge[0])
    soc_next = init_soc * battery_cap + CHARGE_EFFICIENCY * charge0 - (1 / DISCHARGE_EFFICIENCY) * discharge0

    return {
        "grid": grid0,
        "charge": charge0,
        "discharge": discharge0,
        "soc_next": soc_next
    }


def run_rolling_horizon(load, solar, price, battery_cap, charge_power, discharge_power,
                        init_soc, final_soc, horizon_len=3):
    """Run a receding horizon simulation over 24 hours."""
    soc = init_soc * battery_cap
    hourly_costs = []
    grid_schedule = []

    for t in TIME_STEPS:
        end = min(t + horizon_len, HOURS)
        load_window = load[t:end]
        solar_window = solar[t:end]
        price_window = price[t:end]
        final_required = final_soc if end == HOURS else None

        result = solve_horizon_lp(
            load_window,
            solar_window,
            price_window,
            battery_cap,
            charge_power,
            discharge_power,
            soc / battery_cap,
            final_required
        )

        if result is None:
            hourly_costs.append(np.nan)
            grid_schedule.append(0.0)
            continue

        grid_t = result["grid"]
        grid_schedule.append(grid_t)
        hourly_costs.append(grid_t * price[t])
        soc = np.clip(result["soc_next"], 0.0, battery_cap)

    return {
        "hourly_costs": np.array(hourly_costs),
        "total_cost": float(np.nansum(hourly_costs)),
        "grid": np.array(grid_schedule)
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🔋 Microgrid Energy Dispatch Optimization Dashboard")
    st.markdown("**Interactive Linear Programming Solver with Equation-to-Code Mapping**")
    
    # SIDEBAR: CONTROLS
    st.sidebar.markdown("### ⚙️ System Configuration")
    
    battery_cap = st.sidebar.slider(
        "Battery Capacity (kWh)",
        min_value=20.0, max_value=300.0, value=100.0, step=10.0
    )
    
    charge_power = st.sidebar.slider(
        "Max Charge Power (kW)",
        min_value=5.0, max_value=50.0, value=25.0, step=2.5
    )
    
    discharge_power = st.sidebar.slider(
        "Max Discharge Power (kW)",
        min_value=5.0, max_value=50.0, value=25.0, step=2.5
    )
    
    init_soc_pct = st.sidebar.slider("Initial SOC (%)", 0, 100, 20, 5)
    final_soc_pct = st.sidebar.slider("Final SOC (%)", 0, 100, 50, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Scaling")
    
    demand_scale = st.sidebar.slider("Load Demand Scale", 0.5, 2.0, 1.0, 0.1)
    price_scale = st.sidebar.slider("Price Scale", 0.5, 2.0, 1.0, 0.1)

    with st.sidebar.expander("View Full Mathematical Model"):
        st.markdown("**Decision Variables**")
        st.markdown("- $x_s(t)$ = solar output at hour $t$ (kW)")
        st.markdown("- $x_b(t)$ = battery discharge at hour $t$ (kW)")
        st.markdown("- $x_d(t)$ = diesel output at hour $t$ (kW)")

        st.markdown("**Objective Function**")
        st.latex(r"\min \sum_t [c_s\cdot x_s(t) + c_b\cdot x_b(t) + c_d\cdot x_d(t)]")

        st.markdown("**Constraints**")
        st.latex(r"x_s(t) + x_b(t) + x_d(t) = D(t)")
        st.latex(r"0 \le x_i(t) \le X_{i,\max}")
        st.latex(r"SoC(t+1) = SoC(t) - x_b(t)\cdot \Delta t")

        params_df = pd.DataFrame([
            {"Parameter": "Solar cost (c_s)", "Value": f"£{SOLAR_COST_PER_KWH:.2f}/kWh"},
            {"Parameter": "Battery cost (c_b)", "Value": f"£{BATTERY_COST_PER_KWH:.2f}/kWh"},
            {"Parameter": "Diesel cost (c_d)", "Value": f"£{DIESEL_COST_PER_KWH:.2f}/kWh"},
            {"Parameter": "Solar capacity", "Value": f"{SOLAR_CAPACITY_KWP:.1f} kW"},
            {"Parameter": "Battery discharge cap", "Value": f"{discharge_power:.1f} kW"},
            {"Parameter": "Diesel capacity", "Value": f"{MAX_DIESEL_POWER_KW:.1f} kW"},
        ])
        st.markdown("**Current Parameters**")
        st.dataframe(params_df, use_container_width=True)
    
    # SOLVE
    load_adjusted = LOAD_DEMAND * demand_scale
    price_adjusted = GRID_PRICE * price_scale
    init_soc = init_soc_pct / 100.0
    final_soc = final_soc_pct / 100.0
    
    results_opt, prob_obj = build_optimize_microgrid_model(
        load_adjusted, SOLAR_GENERATION, price_adjusted,
        battery_cap, charge_power, discharge_power,
        init_soc=init_soc, final_soc=final_soc
    )
    
    baseline_cost_adj, baseline_grid_adj = compute_baseline_cost(
        load_adjusted, SOLAR_GENERATION, price_adjusted
    )
    
    # TABS
    tab_overview, tab_equations, tab_profiles, tab_analysis, tab_multi, tab_solver, tab_compare, tab_sensitivity, tab_rolling = st.tabs(
        ["📈 System Overview", "⚡ Mathematical Model", "📊 Profiles", "💡 Insights", "🎯 Multi-Objective Optimization", "🧠 Solver Insights", "⚖️ Algorithm Comparison", "🧪 Sensitivity Analysis", "⏳ Rolling Horizon Simulation"]
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: OVERVIEW
    # ──────────────────────────────────────────────────────────────────────────
    
    with tab_overview:
        st.markdown("## System Overview & Results")
        
        if results_opt['status'] == 'Optimal':
            
            opt_cost = results_opt['objective']
            savings = baseline_cost_adj - opt_cost
            savings_pct = (savings / baseline_cost_adj * 100) if baseline_cost_adj > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Optimized Cost", f"£{opt_cost:.2f}", f"-£{savings:.2f}", delta_color="inverse")
            with col2:
                st.metric("Baseline Cost", f"£{baseline_cost_adj:.2f}", "no battery")
            with col3:
                st.metric("Cost Savings", f"£{savings:.2f}", f"{savings_pct:.1f}%")
            with col4:
                peak_load = load_adjusted.max()
                st.metric("Peak Load", f"{peak_load:.1f} kW", f"Avg: {load_adjusted.mean():.1f} kW")
            with col5:
                solar_total = SOLAR_GENERATION.sum()
                st.metric("Solar Generated", f"{solar_total:.1f} kWh", "Daily total")
            
            # Results table
            results_df = pd.DataFrame({
                'Hour': TIME_STEPS,
                'Load_kW': load_adjusted,
                'Solar_kW': SOLAR_GENERATION,
                'Grid_kW': [results_opt['grid'][t] for t in TIME_STEPS],
                'Charge_kW': [results_opt['charge'][t] for t in TIME_STEPS],
                'Discharge_kW': [results_opt['discharge'][t] for t in TIME_STEPS],
                'SoC_kWh': [results_opt['soc'][t] for t in TIME_STEPS],
                'Price_GBP/kWh': price_adjusted,
                'Cost_GBP': price_adjusted * np.array([results_opt['grid'][t] for t in TIME_STEPS]),
            })
            
            st.markdown("### Hourly Dispatch Schedule")
            st.dataframe(results_df, use_container_width=True, height=400)

            st.markdown("### Grid Topology")
            avg_grid = np.mean([results_opt['grid'][t] for t in TIME_STEPS])
            avg_battery = np.mean([results_opt['discharge'][t] for t in TIME_STEPS])
            avg_solar = np.mean(SOLAR_GENERATION)

            flows = {
                "Solar Panel": avg_solar,
                "Battery Storage": avg_battery,
                "Diesel Generator": 0.0,
                "Grid Import": avg_grid,
            }

            metadata = {
                "Solar Panel": {
                    "capacity": SOLAR_CAPACITY_KWP,
                    "output": avg_solar,
                    "cost": SOLAR_COST_PER_KWH,
                },
                "Battery Storage": {
                    "capacity": discharge_power,
                    "output": avg_battery,
                    "cost": BATTERY_COST_PER_KWH,
                },
                "Diesel Generator": {
                    "capacity": MAX_DIESEL_POWER_KW,
                    "output": 0.0,
                    "cost": DIESEL_COST_PER_KWH,
                },
                "Grid Import": {
                    "capacity": "Unlimited",
                    "output": avg_grid,
                    "cost": f"{np.mean(price_adjusted):.2f}",
                },
                "Load Demand": {
                    "capacity": load_adjusted.max(),
                    "output": np.mean(load_adjusted),
                    "cost": "N/A",
                },
            }

            fig_topology = build_grid_topology_plot(flows, metadata)
            st.plotly_chart(fig_topology, use_container_width=True)
            
        else:
            st.error(f"Solver Status: {results_opt['status']}")
            st.warning("Cannot solve with current constraints. Try reducing demand or increasing battery capacity.")
    
    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: MATHEMATICAL MODEL WITH EQUATIONS
    # ──────────────────────────────────────────────────────────────────────────
    
    with tab_equations:
        st.markdown("## 📐 Mathematical Model & Equation-to-Code Mapping")
        
        st.info("Each equation is matched to its Python implementation. This shows how math becomes code.")
        
        # Objective
        st.markdown("### 1️⃣ OBJECTIVE: Minimize Energy Cost")
        st.latex(r"Z = \sum_{t=0}^{23} p_t \times g_t")
        st.markdown("**Where:** $p_t$ = price, $g_t$ = grid purchase")
        st.code("""
prob += pulp.lpSum(price[t] * grid[t] for t in TIME_STEPS), "Total_Energy_Cost"
        """)
        
        st.markdown("---")
        
        # C1: Energy Balance
        st.markdown("### 2️⃣ CONSTRAINT: Energy Balance (Power Conservation)")
        st.latex(r"\text{Load}_t = g_t + d_t + \text{Solar}_t - c_t")
        st.markdown("**Meaning:** Everything in = Everything out (Kirchhoff's law)")
        st.code("""
prob += (grid[t] + discharge[t] + solar[t] == load[t] + charge[t], f"energy_balance_{t}")
        """)
        
        st.markdown("---")
        
        # C2: SOC dynamics
        st.markdown("### 3️⃣ CONSTRAINT: Battery State Dynamics (Physics)")
        st.latex(r"b_{t+1} = b_t + \eta_c \cdot c_t - \frac{1}{\eta_d} \cdot d_t")
        st.markdown("**Meaning:** Battery level changes based on charge in / discharge out minus losses")
        st.code("""
prob += (soc[t+1] == soc[t] + charge_eff * charge[t] - (1/discharge_eff) * discharge[t])
        """)
        
        st.markdown("---")
        
        # C3: Capacity
        st.markdown("### 4️⃣ CONSTRAINT: Battery Capacity Limits")
        st.latex(r"0 \le b_t \le B_{\max}")
        st.markdown("**Meaning:** Stored energy cannot exceed tank size")
        st.code("""
soc[t] = pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=battery_cap, cat='Continuous')
        """)
        
        st.markdown("---")
        
        # C4-5: Power limits
        st.markdown("### 5️⃣ CONSTRAINT: Power Rate Limits")
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.latex(r"c_t \le P_{c,\max}")
            st.code("prob += (charge[t] <= charge_power)")
        with col_eq2:
            st.latex(r"d_t \le P_{d,\max}")
            st.code("prob += (discharge[t] <= discharge_power)")
        
        st.markdown("---")
        
        # C6-7: Boundary
        st.markdown("### 6️⃣ CONSTRAINTS: Boundary Conditions")
        col_bc1, col_bc2 = st.columns(2)
        with col_bc1:
            st.markdown("**Initial:**")
            st.latex(r"b_0 = \text{init\_soc} \cdot B_{\max}")
        with col_bc2:
            st.markdown("**Terminal:**")
            st.latex(r"b_{23} \ge \text{final\_soc} \cdot B_{\max}")
        
        st.markdown("---")
        st.markdown("### Problem Summary")
        st.write(f"- **Time periods:** 24 hours")
        st.write(f"- **Decision variables:** 72 (4 variables × 24 hours)")
        st.write(f"- **Constraints:** ~105 (energy balance + physics + limits)")
        st.write(f"- **Type:** Linear Program (LP)")
        st.write(f"- **Solver:** CBC (Coin-or-branch-and-cut)")
        st.write(f"- **Optimality:** Guarantees global optimum")
    
    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: PROFILES & ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    
    with tab_profiles:
        st.markdown("## Hourly Profiles & Detailed Analysis")
        
        if results_opt['status'] == 'Optimal':
            
            grid_opt = np.array([results_opt['grid'][t] for t in TIME_STEPS])
            charge_opt = np.array([results_opt['charge'][t] for t in TIME_STEPS])
            discharge_opt = np.array([results_opt['discharge'][t] for t in TIME_STEPS])
            soc_opt = np.array([results_opt['soc'][t] for t in TIME_STEPS])
            
            # Plot 1: Load, Solar, Grid
            fig1, ax1 = plt.subplots(figsize=(14, 5))
            ax1.plot(TIME_STEPS, load_adjusted, 'o-', label='Load Demand', linewidth=2.5, color='#d62728')
            ax1.plot(TIME_STEPS, SOLAR_GENERATION, 's-', label='Solar Generation', linewidth=2.5, color='#ff7f0e')
            ax1.plot(TIME_STEPS, grid_opt, '^-', label='Grid Purchase (Optimized)', linewidth=2, color='#1f77b4')
            ax1.fill_between(TIME_STEPS, 0, load_adjusted, alpha=0.1, color='red')
            ax1.set_xlabel('Hour', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
            ax1.set_title('Load vs Solar vs Grid Purchase', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
            # Plot 2: Battery
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.bar(TIME_STEPS, charge_opt, label='Charging', color='#2ca02c', alpha=0.7, width=0.8)
            ax2.bar(TIME_STEPS, -discharge_opt, label='Discharging', color='#d62728', alpha=0.7, width=0.8)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_xlabel('Hour', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
            ax2.set_title('Battery Charge/Discharge Schedule', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig2)
            
            # Plot 3: SOC
            fig3, ax3 = plt.subplots(figsize=(14, 5))
            ax3.fill_between(TIME_STEPS, 0, soc_opt, alpha=0.3, color='#1f77b4')
            ax3.plot(TIME_STEPS, soc_opt, 'o-', color='#1f77b4', linewidth=2.5, markersize=6)
            ax3.axhline(battery_cap, color='gray', linestyle='--', linewidth=1, label=f'Max ({battery_cap} kWh)')
            ax3.axhline(init_soc * battery_cap, color='orange', linestyle='--', alpha=0.7, label=f'Init ({init_soc*100:.0f}%)')
            ax3.axhline(final_soc * battery_cap, color='red', linestyle='--', alpha=0.7, label=f'Min ({final_soc*100:.0f}%)')
            ax3.set_xlabel('Hour', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Energy (kWh)', fontsize=11, fontweight='bold')
            ax3.set_title('Battery State of Charge', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            
            # Plot 4: Cost
            cost_opt = price_adjusted * grid_opt
            cost_base = price_adjusted * baseline_grid_adj
            
            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
            
            x = np.arange(len(TIME_STEPS))
            w = 0.35
            ax4a.bar(x - w/2, cost_base, w, label='Baseline', color='#d62728', alpha=0.7)
            ax4a.bar(x + w/2, cost_opt, w, label='Optimized', color='#2ca02c', alpha=0.7)
            ax4a.set_xlabel('Hour', fontsize=10, fontweight='bold')
            ax4a.set_ylabel('Cost (£)', fontsize=10, fontweight='bold')
            ax4a.set_xticks(x)
            ax4a.set_xticklabels(range(24))
            ax4a.legend(fontsize=9)
            ax4a.grid(True, alpha=0.3, axis='y')
            ax4a.set_title('Hourly Cost Comparison', fontsize=11, fontweight='bold')
            
            opt_cost = results_opt['objective']
            categories = ['Baseline', 'Optimized']
            costs = [baseline_cost_adj, opt_cost]
            colors = ['#d62728', '#2ca02c']
            ax4b.bar(categories, costs, color=colors, alpha=0.7, edgecolor='black')
            for i, (cat, cost) in enumerate(zip(categories, costs)):
                ax4b.text(i, cost + 1, f'£{cost:.2f}', ha='center', fontweight='bold')
            ax4b.set_ylabel('Daily Cost (£)', fontsize=10, fontweight='bold')
            ax4b.set_title(f'Daily Total (Save: £{baseline_cost_adj - opt_cost:.2f})', fontsize=11, fontweight='bold')
            st.pyplot(fig4)
            
            # Metrics
            st.markdown("### Key Metrics")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.subheader("Energy")
                total_load = load_adjusted.sum()
                total_solar = SOLAR_GENERATION.sum()
                st.write(f"**Daily Load:** {total_load:.1f} kWh")
                st.write(f"**Solar Gen:** {total_solar:.1f} kWh")
                st.write(f"**Grid (Opt):** {grid_opt.sum():.1f} kWh")
                st.write(f"**Grid (Base):** {baseline_grid_adj.sum():.1f} kWh")
            
            with col_m2:
                st.subheader("Battery")
                st.write(f"**Charged:** {charge_opt.sum():.1f} kWh")
                st.write(f"**Discharged:** {discharge_opt.sum():.1f} kWh")
                st.write(f"**Cycles:** {charge_opt.sum() / battery_cap:.2f}")
                st.write(f"**Avg SOC:** {soc_opt.mean():.1f} kWh")
            
            with col_m3:
                st.subheader("Cost & Savings")
                opt_cost = results_opt['objective']
                savings = baseline_cost_adj - opt_cost
                st.write(f"**Daily (Opt):** £{opt_cost:.2f}")
                st.write(f"**Daily (Base):** £{baseline_cost_adj:.2f}")
                st.write(f"**Saving:** £{savings:.2f}")
                st.write(f"**Annual:** £{savings*365:.0f}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: INSIGHTS
    # ──────────────────────────────────────────────────────────────────────────
    
    with tab_analysis:
        st.markdown("## Business Insights & ROI")
        
        if results_opt['status'] == 'Optimal':
            
            opt_cost = results_opt['objective']
            savings = baseline_cost_adj - opt_cost
            savings_pct = (savings / baseline_cost_adj * 100) if baseline_cost_adj > 0 else 0
            
            grid_opt = np.array([results_opt['grid'][t] for t in TIME_STEPS])
            charge_opt = np.array([results_opt['charge'][t] for t in TIME_STEPS])
            discharge_opt = np.array([results_opt['discharge'][t] for t in TIME_STEPS])
            soc_opt = np.array([results_opt['soc'][t] for t in TIME_STEPS])
            
            st.subheader("🎯 Key Findings")
            
            # Finding 1: Peak Shaving
            peak_load = load_adjusted.max()
            grid_peak = max(grid_opt)
            st.write(f"""
**1. Peak Load Reduction**
- System peak load: {peak_load:.1f} kW
- Peak grid purchase: {grid_peak:.1f} kW
- Reduction: {peak_load - grid_peak:.1f} kW ({(peak_load - grid_peak)/peak_load*100:.1f}%)
- Annual demand charge savings: £{(peak_load - grid_peak) * 30 * 12:.0f}
            """)
            
            # Finding 2: Load Shifting
            charging_hours = [t for t in TIME_STEPS if charge_opt[t] > 0.5]
            discharging_hours = [t for t in TIME_STEPS if discharge_opt[t] > 0.5]
            avg_charge_price = price_adjusted[charging_hours].mean() if charging_hours else 0
            avg_discharge_price = price_adjusted[discharging_hours].mean() if discharging_hours else 0
            
            st.write(f"""
**2. Load Shifting & Price Arbitrage**
- Charges during {charging_hours} at £{avg_charge_price:.3f}/kWh (avg)
- Discharges during {discharging_hours} at £{avg_discharge_price:.3f}/kWh (avg)
- Price difference exploited: £{avg_discharge_price - avg_charge_price:.3f}/kWh
- Daily arbitrage value: £{savings:.2f}
            """)
            
            # Finding 3: ROI
            battery_cost_per_kwh = 400
            battery_total_cost = battery_cap * battery_cost_per_kwh
            annual_savings_total = savings * 365
            payback = battery_total_cost / annual_savings_total if annual_savings_total > 0 else float('inf')
            
            st.write(f"""
**3. Battery Economics**
- Battery capex: £{battery_total_cost:,.0f} (£{battery_cost_per_kwh}/kWh)
- Annual energy savings: £{annual_savings_total:,.0f}
- **Payback period: {payback:.1f} years**
- Battery life: 10-15 years → Highly profitable
            """)
            
            # Summary
            st.info(f"""
**BUSINESS CASE SUMMARY**

✓ Daily savings: £{savings:.2f} ({savings_pct:.1f}%)
✓ Annual savings: £{annual_savings_total:,.0f}
✓ 3-year savings: £{annual_savings_total*3:,.0f}
✓ Payback: {payback:.1f} years
✓ 10-year NPV (no discount): £{annual_savings_total*10 - battery_total_cost:,.0f}

**Battery is economically justified.**
            """)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6: SOLVER INSIGHTS
    # ──────────────────────────────────────────────────────────────────────────

    with tab_solver:
        st.markdown("## Solver Insights")
        st.markdown("Inspect dual values and binding constraints from the LP solution.")

        with st.expander("What are shadow prices?", expanded=False):
            st.write(
                "Shadow prices (dual values) show how the objective would change if a constraint "
                "were relaxed by 1 kW. For a microgrid operator, this indicates the marginal value "
                "of extra capacity or flexibility at the optimum."
            )

        if results_opt['status'] != 'Optimal':
            st.warning("Solver insights are available only for an optimal solution.")
        else:
            constraints = prob_obj.constraints
            insight_rows = []
            binding_rows = []
            tol = 1e-6

            for name, constraint in constraints.items():
                dual = getattr(constraint, "pi", None)
                slack = getattr(constraint, "slack", None)
                is_binding = False
                if slack is not None:
                    is_binding = abs(slack) <= tol

                insight_rows.append({
                    "Constraint": name,
                    "Shadow Price": dual,
                    "Slack": slack,
                    "Binding": is_binding
                })

                if is_binding:
                    savings = None if dual is None else -dual
                    binding_rows.append({
                        "Constraint": name,
                        "Shadow Price": dual,
                        "Relaxing by 1 kW saves (₹/hour)": savings
                    })

            st.markdown("### Shadow Prices (Dual Variables)")
            st.dataframe(pd.DataFrame(insight_rows), use_container_width=True)

            st.markdown("### Binding Constraints")
            st.dataframe(pd.DataFrame(binding_rows), use_container_width=True)

            st.markdown("### Simplex Iterations")
            st.write("Not available from CBC via PuLP. Use a solver that exposes iterations if needed.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7: ALGORITHM COMPARISON
    # ──────────────────────────────────────────────────────────────────────────

    with tab_compare:
        st.markdown("## Algorithm Comparison")
        st.markdown("Compare greedy dispatch vs. genetic algorithm optimization on the same profile.")

        if st.button("Run Comparison"):
            cost_factors = {
                "solar": SOLAR_COST_PER_KWH,
                "battery": BATTERY_COST_PER_KWH,
                "diesel": DIESEL_COST_PER_KWH,
            }
            emission_factors = {
                "solar": SOLAR_EMISSIONS_KG_PER_KWH,
                "battery": BATTERY_EMISSIONS_KG_PER_KWH,
                "diesel": DIESEL_EMISSIONS_KG_PER_KWH,
            }

            greedy_dispatch = simulate_dispatch(
                load_adjusted,
                SOLAR_GENERATION,
                battery_cap,
                charge_power,
                discharge_power,
                init_soc,
                final_soc,
                MAX_DIESEL_POWER_KW
            )
            greedy_cost, greedy_emissions = evaluate_dispatch_costs(
                greedy_dispatch, cost_factors, emission_factors
            )

            ga_dispatch, ga_cost, ga_emissions, ga_history = run_ga_dispatch(
                load_adjusted,
                SOLAR_GENERATION,
                battery_cap,
                charge_power,
                discharge_power,
                init_soc,
                final_soc,
                MAX_DIESEL_POWER_KW,
                cost_factors,
                emission_factors
            )

            st.session_state["compare_results"] = {
                "greedy": {
                    "dispatch": greedy_dispatch,
                    "cost": greedy_cost,
                    "emissions": greedy_emissions,
                    "reliability": greedy_dispatch["reliability"],
                },
                "ga": {
                    "dispatch": ga_dispatch,
                    "cost": ga_cost,
                    "emissions": ga_emissions,
                    "reliability": ga_dispatch["reliability"],
                    "history": ga_history,
                },
            }

        compare_results = st.session_state.get("compare_results")
        if not compare_results:
            st.info("Click 'Run Comparison' to execute the algorithms.")
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                st.metric("Greedy Total Cost", f"£{compare_results['greedy']['cost']:.2f}")
            with col_right:
                st.metric("GA Total Cost", f"£{compare_results['ga']['cost']:.2f}")

            fig_greedy = go.Figure()
            fig_greedy.add_trace(go.Bar(
                x=TIME_STEPS,
                y=compare_results["greedy"]["dispatch"]["solar_use"],
                name="Solar",
                marker_color="#ff7f0e"
            ))
            fig_greedy.add_trace(go.Bar(
                x=TIME_STEPS,
                y=compare_results["greedy"]["dispatch"]["battery_discharge"],
                name="Battery",
                marker_color="#2ca02c"
            ))
            fig_greedy.add_trace(go.Bar(
                x=TIME_STEPS,
                y=compare_results["greedy"]["dispatch"]["diesel_use"],
                name="Diesel",
                marker_color="#d62728"
            ))
            fig_greedy.update_layout(
                barmode="stack",
                title="Greedy Dispatch (kW)",
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                height=350
            )
            st.plotly_chart(fig_greedy, use_container_width=True)

            fig_ga = go.Figure()
            fig_ga.add_trace(go.Bar(
                x=TIME_STEPS,
                y=compare_results["ga"]["dispatch"]["solar_use"],
                name="Solar",
                marker_color="#ff7f0e"
            ))
            fig_ga.add_trace(go.Bar(
                x=TIME_STEPS,
                y=compare_results["ga"]["dispatch"]["battery_discharge"],
                name="Battery",
                marker_color="#2ca02c"
            ))
            fig_ga.add_trace(go.Bar(
                x=TIME_STEPS,
                y=compare_results["ga"]["dispatch"]["diesel_use"],
                name="Diesel",
                marker_color="#d62728"
            ))
            fig_ga.update_layout(
                barmode="stack",
                title="GA Dispatch (kW)",
                xaxis_title="Hour",
                yaxis_title="Power (kW)",
                height=350
            )
            st.plotly_chart(fig_ga, use_container_width=True)

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(1, len(compare_results["ga"]["history"]) + 1)),
                y=compare_results["ga"]["history"],
                mode="lines+markers",
                name="Best Fitness",
                line=dict(color="#1f77b4")
            ))
            fig_conv.update_layout(
                title="GA Convergence Curve",
                xaxis_title="Generation",
                yaxis_title="Best Fitness (Cost + Penalty)",
                height=350
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            summary_df = pd.DataFrame([
                {
                    "Method": "Greedy",
                    "Cost": compare_results["greedy"]["cost"],
                    "Emissions": compare_results["greedy"]["emissions"],
                    "Reliability (%)": compare_results["greedy"]["reliability"],
                },
                {
                    "Method": "Genetic Algorithm",
                    "Cost": compare_results["ga"]["cost"],
                    "Emissions": compare_results["ga"]["emissions"],
                    "Reliability (%)": compare_results["ga"]["reliability"],
                },
            ])
            st.markdown("### Summary")
            st.dataframe(summary_df, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 8: SENSITIVITY ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────

    with tab_sensitivity:
        st.markdown("## What-If Decision Tool")
        st.write("Used by energy managers to stress-test dispatch plans.")

        solar_change = st.slider(
            "Solar Capacity Change (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5
        )
        diesel_change = st.slider(
            "Diesel Cost Change (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5
        )

        base_cost_factors = {
            "solar": SOLAR_COST_PER_KWH,
            "battery": BATTERY_COST_PER_KWH,
            "diesel": DIESEL_COST_PER_KWH,
        }
        base_emission_factors = {
            "solar": SOLAR_EMISSIONS_KG_PER_KWH,
            "battery": BATTERY_EMISSIONS_KG_PER_KWH,
            "diesel": DIESEL_EMISSIONS_KG_PER_KWH,
        }

        base_results = solve_multiobjective_dispatch(
            load_adjusted,
            SOLAR_GENERATION,
            battery_cap,
            charge_power,
            discharge_power,
            init_soc,
            final_soc,
            1.0,
            MAX_DIESEL_POWER_KW,
            base_cost_factors,
            base_emission_factors
        )

        solar_scale = 1.0 + (solar_change / 100.0)
        diesel_scale = 1.0 + (diesel_change / 100.0)
        adjusted_solar = SOLAR_GENERATION * solar_scale
        adjusted_cost_factors = {
            "solar": SOLAR_COST_PER_KWH,
            "battery": BATTERY_COST_PER_KWH,
            "diesel": DIESEL_COST_PER_KWH * diesel_scale,
        }

        new_results = solve_multiobjective_dispatch(
            load_adjusted,
            adjusted_solar,
            battery_cap,
            charge_power,
            discharge_power,
            init_soc,
            final_soc,
            1.0,
            MAX_DIESEL_POWER_KW,
            adjusted_cost_factors,
            base_emission_factors
        )

        if base_results["status"] != "Optimal" or new_results["status"] != "Optimal":
            st.warning("Sensitivity results unavailable due to non-optimal solution.")
        else:
            base_cost = base_results["cost"]
            new_cost = new_results["cost"]
            delta_cost = new_cost - base_cost
            delta_pct = (delta_cost / base_cost * 100) if base_cost else 0.0

            st.metric(
                "Optimal Cost Change",
                f"₹{new_cost:.2f}",
                f"₹{delta_cost:.2f} ({delta_pct:.1f}%)"
            )

            base_solar = np.sum([base_results["solar_use"][t] for t in TIME_STEPS])
            base_battery = np.sum([base_results["discharge"][t] for t in TIME_STEPS])
            base_diesel = np.sum([base_results["diesel"][t] for t in TIME_STEPS])
            total_supply = base_solar + base_battery + base_diesel

            fig_pie = go.Figure(data=[go.Pie(
                labels=["Solar", "Battery", "Diesel"],
                values=[base_solar, base_battery, base_diesel],
                hole=0.4
            )])
            fig_pie.update_layout(title="New Dispatch Mix (Energy Share)")
            st.plotly_chart(fig_pie, use_container_width=True)

            sensitivity_df = pd.DataFrame([
                {
                    "Parameter Changed": "Solar Capacity",
                    "Original Value": f"{SOLAR_CAPACITY_KWP:.1f} kWp",
                    "New Value": f"{SOLAR_CAPACITY_KWP * solar_scale:.1f} kWp",
                    "Cost Impact (₹)": f"{delta_cost:.2f}",
                },
                {
                    "Parameter Changed": "Diesel Cost",
                    "Original Value": f"£{DIESEL_COST_PER_KWH:.2f}/kWh",
                    "New Value": f"£{DIESEL_COST_PER_KWH * diesel_scale:.2f}/kWh",
                    "Cost Impact (₹)": f"{delta_cost:.2f}",
                },
            ])
            st.markdown("### Sensitivity Table")
            st.dataframe(sensitivity_df, use_container_width=True)

            tornado_values = [
                ("Solar Capacity", abs(delta_cost)),
                ("Diesel Cost", abs(delta_cost)),
            ]
            tornado_df = pd.DataFrame(tornado_values, columns=["Parameter", "Impact"])
            fig_tornado = go.Figure(go.Bar(
                x=tornado_df["Impact"],
                y=tornado_df["Parameter"],
                orientation="h",
                marker_color="#1f77b4"
            ))
            fig_tornado.update_layout(
                title="Tornado Chart: Cost Impact",
                xaxis_title="Absolute Cost Impact (₹)",
                yaxis_title="Parameter",
                height=300
            )
            st.plotly_chart(fig_tornado, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 9: ROLLING HORIZON SIMULATION
    # ──────────────────────────────────────────────────────────────────────────

    with tab_rolling:
        st.markdown("## Rolling Horizon Simulation")
        st.text_area(
            "Explanation",
            value="Real-world energy systems use rolling horizon because future demand is uncertain.",
            height=80
        )

        if results_opt['status'] != 'Optimal':
            st.warning("Rolling horizon comparison requires an optimal baseline solution.")
        else:
            perfect_hourly = price_adjusted * np.array([results_opt['grid'][t] for t in TIME_STEPS])
            perfect_cost = results_opt['objective']

            rolling_results = run_rolling_horizon(
                load_adjusted,
                SOLAR_GENERATION,
                price_adjusted,
                battery_cap,
                charge_power,
                discharge_power,
                init_soc,
                final_soc,
                horizon_len=3
            )

            rolling_cost = rolling_results["total_cost"]
            gap_pct = ((rolling_cost - perfect_cost) / perfect_cost * 100) if perfect_cost else 0.0

            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=TIME_STEPS,
                y=perfect_hourly,
                mode="lines+markers",
                name="Perfect Foresight",
                line=dict(color="#1f77b4")
            ))
            fig_roll.add_trace(go.Scatter(
                x=TIME_STEPS,
                y=rolling_results["hourly_costs"],
                mode="lines+markers",
                name="Rolling Horizon",
                line=dict(color="#ff7f0e")
            ))
            fig_roll.update_layout(
                title="Hourly Cost Comparison",
                xaxis_title="Hour",
                yaxis_title="Cost (£)",
                height=350
            )
            st.plotly_chart(fig_roll, use_container_width=True)

            summary = pd.DataFrame([
                {"Method": "Perfect Foresight", "Total Cost": perfect_cost},
                {"Method": "Rolling Horizon", "Total Cost": rolling_cost},
                {"Method": "Optimality Gap", "Total Cost": f"{gap_pct:.2f}%"},
            ])
            st.markdown("### Total Daily Cost")
            st.dataframe(summary, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5: MULTI-OBJECTIVE OPTIMIZATION
    # ──────────────────────────────────────────────────────────────────────────

    with tab_multi:
        st.markdown("## Multi-Objective Optimization")
        st.markdown("Explore the trade-off between cost and emissions using a weighted objective.")

        alpha = st.slider(
            "Cost vs. Green Priority",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

        cost_factors = {
            "solar": SOLAR_COST_PER_KWH,
            "battery": BATTERY_COST_PER_KWH,
            "diesel": DIESEL_COST_PER_KWH,
        }
        emission_factors = {
            "solar": SOLAR_EMISSIONS_KG_PER_KWH,
            "battery": BATTERY_EMISSIONS_KG_PER_KWH,
            "diesel": DIESEL_EMISSIONS_KG_PER_KWH,
        }

        pareto_alphas = np.linspace(0, 1, 20)
        pareto_costs = []
        pareto_emissions = []

        for a in pareto_alphas:
            res = solve_multiobjective_dispatch(
                load_adjusted,
                SOLAR_GENERATION,
                battery_cap,
                charge_power,
                discharge_power,
                init_soc,
                final_soc,
                float(a),
                MAX_DIESEL_POWER_KW,
                cost_factors,
                emission_factors
            )
            pareto_costs.append(res["cost"])
            pareto_emissions.append(res["emissions"])

        selected = solve_multiobjective_dispatch(
            load_adjusted,
            SOLAR_GENERATION,
            battery_cap,
            charge_power,
            discharge_power,
            init_soc,
            final_soc,
            alpha,
            MAX_DIESEL_POWER_KW,
            cost_factors,
            emission_factors
        )

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(
            x=pareto_emissions,
            y=pareto_costs,
            mode="markers+lines",
            name="Pareto Front",
            marker=dict(color="#1f77b4", size=8),
            hovertemplate="Emissions: %{x:.2f} kg<br>Cost: £%{y:.2f}<extra></extra>"
        ))
        fig_pareto.add_trace(go.Scatter(
            x=[selected["emissions"]],
            y=[selected["cost"]],
            mode="markers",
            name="Selected alpha",
            marker=dict(color="red", size=12),
            hovertemplate="Selected<br>Emissions: %{x:.2f} kg<br>Cost: £%{y:.2f}<extra></extra>"
        ))
        fig_pareto.update_layout(
            title="Cost vs Emissions Trade-off",
            xaxis_title="Total Emissions (kg CO2)",
            yaxis_title="Total Cost (£)",
            height=450
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        if selected["status"] != "Optimal":
            st.error(f"Solver Status: {selected['status']}")
        else:
            total_solar = np.sum([selected["solar_use"][t] for t in TIME_STEPS])
            total_battery = np.sum([selected["discharge"][t] for t in TIME_STEPS])
            total_diesel = np.sum([selected["diesel"][t] for t in TIME_STEPS])

            fig_dispatch = go.Figure(data=[
                go.Bar(
                    x=["Solar", "Battery", "Diesel"],
                    y=[total_solar, total_battery, total_diesel],
                    marker_color=["#ff7f0e", "#2ca02c", "#d62728"]
                )
            ])
            fig_dispatch.update_layout(
                title="Optimal Dispatch Breakdown (kWh)",
                xaxis_title="Source",
                yaxis_title="Energy (kWh)",
                height=350
            )
            st.plotly_chart(fig_dispatch, use_container_width=True)


if __name__ == "__main__":
    main()
