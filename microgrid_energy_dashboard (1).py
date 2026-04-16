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
    tab_overview, tab_equations, tab_profiles, tab_analysis = st.tabs(
        ["📈 System Overview", "⚡ Mathematical Model", "📊 Profiles", "💡 Insights"]
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


if __name__ == "__main__":
    main()
