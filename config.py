from pathlib import Path

# 项目路径
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_TABLE_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURE_DIR = BASE_DIR / "outputs" / "figures"

# 输入文件
DISTANCE_FILE = DATA_DIR / "distances.csv"
Q1_DEMAND_FILE = DATA_DIR / "peopleQ1.csv"

# 输出文件
ROUTES_CSV = OUTPUT_TABLE_DIR / "q1-routes.csv"
ASSIGNMENTS_CSV = OUTPUT_TABLE_DIR / "q1-assignments.csv"
ROUTES_XLSX = OUTPUT_TABLE_DIR / "q1-routes.xlsx"
ASSIGNMENTS_XLSX = OUTPUT_TABLE_DIR / "q1-assignments.xlsx"
SUMMARY_XLSX = OUTPUT_TABLE_DIR / "q1-summary.xlsx"

ALNS_CONVERGENCE_PNG = OUTPUT_FIGURE_DIR / "q1_alns_convergence.png"
METHOD_COMPARE_PNG = OUTPUT_FIGURE_DIR / "q1_method_comparison.png"
NETWORK_PNG = OUTPUT_FIGURE_DIR / "q1_route_network.png"
STOP_COUNT_PNG = OUTPUT_FIGURE_DIR / "q1_stop_count_distribution.png"
SEED_COMPARE_PNG = OUTPUT_FIGURE_DIR / "q1_seed_comparison.png"

# 题目常量
AIRPORTS = ("A01", "A02", "A03")
GAS_STATIONS = ("F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050")
GAS_SET = set(GAS_STATIONS)

AIRCRAFT = {
    "T1": {"seats": 12, "speed": 250.0, "burn": 3.4, "tank": 1000.0, "reserve": 150.0},
    "T2": {"seats": 16, "speed": 220.0, "burn": 2.5, "tank": 1150.0, "reserve": 150.0},
    "T3": {"seats": 19, "speed": 190.0, "burn": 2.9, "tank": 1600.0, "reserve": 200.0},
}

MAX_LANDINGS = 5
MAX_SEATS = max(item["seats"] for item in AIRCRAFT.values())
EPS = 1e-9

# 求解参数
SAVINGS_RANDOM_TOP_K = 6
SAVINGS_MULTI_STARTS = 3
ALNS_ITERATIONS = 36
ALNS_REMOVE_RATIO = 0.18
ALNS_INITIAL_TEMPERATURE = 120.0
ALNS_COOLING_RATE = 0.985
ALNS_REWARD_GLOBAL = 6.0
ALNS_REWARD_CURRENT = 3.0
ALNS_REWARD_ACCEPT = 0.8
ALNS_WEIGHT_DECAY = 0.2
MULTI_SEEDS = (20260814, 20260815, 20260816)
RANDOM_SEED = 20260814
