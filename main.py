#!/usr/bin/env python3

import os
import time
import math
import json
import signal
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque


RUNNING = True


def signal_handler(_sig, _frame) -> None:
    global RUNNING
    RUNNING = False


@dataclass
class PacketSnapshot:
    timestamp: float
    rx_packets: int
    tx_packets: int


@dataclass
class PacketSample:
    timestamp: float
    rx_delta: int
    tx_delta: int
    rx_pps: float
    tx_pps: float
    total_delta: int
    total_pps: float


@dataclass
class ForecastResult:
    value: float
    lower: float
    upper: float
    residual: float
    anomaly_score: float
    is_anomaly: bool


@dataclass
class SeriesStats:
    count: int = 0
    minimum: float = 0.0
    maximum: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0
    variance: float = 0.0
    p95: float = 0.0
    last: float = 0.0
    trend: float = 0.0


@dataclass
class ModelConfig:
    interval: float = 2.0
    window_size: int = 120
    ar_order: int = 5
    diff_order: int = 0
    seasonal_period: int = 0
    anomaly_sigma: float = 3.0
    warmup_samples: int = 20
    forecast_horizon: int = 1
    ewma_alpha: float = 0.25
    min_interval: float = 0.2


@dataclass
class InterfaceModelState:
    iface: str
    config: ModelConfig
    snapshots: Deque[PacketSnapshot] = field(default_factory=deque)
    samples: Deque[PacketSample] = field(default_factory=deque)
    total_series: Deque[float] = field(default_factory=deque)
    rx_series: Deque[float] = field(default_factory=deque)
    tx_series: Deque[float] = field(default_factory=deque)
    residual_series: Deque[float] = field(default_factory=deque)
    ewma_total: Optional[float] = None
    ewma_rx: Optional[float] = None
    ewma_tx: Optional[float] = None
    last_forecast: Optional[ForecastResult] = None

    def __post_init__(self) -> None:
        self.snapshots = deque(maxlen=self.config.window_size + 2)
        self.samples = deque(maxlen=self.config.window_size)
        self.total_series = deque(maxlen=self.config.window_size)
        self.rx_series = deque(maxlen=self.config.window_size)
        self.tx_series = deque(maxlen=self.config.window_size)
        self.residual_series = deque(maxlen=self.config.window_size)


def read_proc_net_dev() -> Dict[str, Dict[str, int]]:
    data: Dict[str, Dict[str, int]] = {}
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return data

    for line in lines[2:]:
        if ":" not in line:
            continue
        name, stats = line.split(":", 1)
        iface = name.strip()
        parts = stats.split()
        if len(parts) < 16:
            continue
        try:
            data[iface] = {
                "rx_packets": int(parts[1]),
                "tx_packets": int(parts[9]),
            }
        except ValueError:
            continue
    return data


def difference_series(values: List[float], order: int) -> List[float]:
    result = values[:]
    for _ in range(order):
        if len(result) < 2:
            return []
        result = [result[i] - result[i - 1] for i in range(1, len(result))]
    return result


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: List[float], sample: bool = True) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    denom = n - 1 if sample else n
    return sum((x - m) ** 2 for x in values) / denom


def stddev(values: List[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def linear_trend(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = mean(xs)
    my = mean(values)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def autocovariance(values: List[float], lag: int) -> float:
    n = len(values)
    if n == 0 or lag >= n:
        return 0.0
    m = mean(values)
    return sum((values[t] - m) * (values[t - lag] - m) for t in range(lag, n)) / n


def build_toeplitz_row(autocovs: List[float], row: int, size: int) -> List[float]:
    return [autocovs[abs(row - col)] for col in range(size)]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    if n == 0:
        return []
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return [0.0] * n
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def fit_yule_walker_ar(values: List[float], order: int) -> List[float]:
    if order <= 0 or len(values) <= order:
        return []
    autocovs = [autocovariance(values, lag) for lag in range(order + 1)]
    matrix = [build_toeplitz_row(autocovs, row, order) for row in range(order)]
    vector = autocovs[1:]
    coeffs = solve_linear_system(matrix, vector)
    return coeffs


def ar_predict_next(values: List[float], coeffs: List[float]) -> float:
    order = len(coeffs)
    if order == 0 or len(values) < order:
        return values[-1] if values else 0.0
    history = values[-order:]
    return sum(c * x for c, x in zip(coeffs, reversed(history)))


def invert_differences(
    original_values: List[float],
    differenced_forecast: float,
    diff_order: int
) -> float:
    if diff_order <= 0 or not original_values:
        return differenced_forecast
    if diff_order == 1:
        return original_values[-1] + differenced_forecast
    restored = differenced_forecast
    temp = original_values[:]
    for _ in range(diff_order):
        if len(temp) < 2:
            break
        last_diff = temp[-1] - temp[-2]
        restored = temp[-1] + restored
        temp = difference_series(temp, 1)
        if temp:
            restored += last_diff
    return restored


def seasonal_baseline(values: List[float], period: int) -> float:
    if period <= 0 or len(values) < period:
        return mean(values) if values else 0.0
    matching = values[-period::period]
    return mean(matching) if matching else mean(values)


def compute_stats(values: List[float]) -> SeriesStats:
    if not values:
        return SeriesStats()
    count = len(values)
    mn = min(values)
    mx = max(values)
    m = mean(values)
    med = statistics.median(values)
    var = variance(values)
    sd = math.sqrt(var)
    p95 = percentile(values, 95)
    last = values[-1]
    trend = linear_trend(values)
    return SeriesStats(
        count=count,
        minimum=mn,
        maximum=mx,
        mean=m,
        median=med,
        stdev=sd,
        variance=var,
        p95=p95,
        last=last,
        trend=trend,
    )


def zscore(value: float, values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    sd = stddev(values)
    if sd <= 1e-12:
        return 0.0
    m = mean(values)
    return (value - m) / sd


def update_ewma(previous: Optional[float], value: float, alpha: float) -> float:
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def detect_anomaly(
    observed: float,
    predicted: float,
    residuals: List[float],
    sigma_threshold: float
) -> Tuple[float, bool]:
    residual = observed - predicted
    score = abs(zscore(residual, residuals)) if len(residuals) >= 5 else 0.0
    if len(residuals) < 5:
        return score, False
    sd = stddev(residuals)
    if sd <= 1e-12:
        return 0.0, False
    return score, abs(residual - mean(residuals)) > sigma_threshold * sd


def sample_interface(iface: str) -> Optional[PacketSnapshot]:
    data = read_proc_net_dev()
    entry = data.get(iface)
    if entry is None:
        return None
    return PacketSnapshot(
        timestamp=time.time(),
        rx_packets=entry["rx_packets"],
        tx_packets=entry["tx_packets"],
    )


def build_sample(prev: PacketSnapshot, curr: PacketSnapshot) -> Optional[PacketSample]:
    dt = curr.timestamp - prev.timestamp
    if dt <= 0:
        return None
    rx_delta = curr.rx_packets - prev.rx_packets
    tx_delta = curr.tx_packets - prev.tx_packets
    if rx_delta < 0 or tx_delta < 0:
        return None
    total_delta = rx_delta + tx_delta
    return PacketSample(
        timestamp=curr.timestamp,
        rx_delta=rx_delta,
        tx_delta=tx_delta,
        rx_pps=rx_delta / dt,
        tx_pps=tx_delta / dt,
        total_delta=total_delta,
        total_pps=total_delta / dt,
    )


def fit_forecast_for_series(
    original_series: List[float],
    config: ModelConfig,
    residual_history: List[float]
) -> ForecastResult:
    if not original_series:
        return ForecastResult(0.0, 0.0, 0.0, 0.0, 0.0, False)

    prepared = difference_series(original_series, config.diff_order)
    working = prepared if prepared else original_series
    order = min(config.ar_order, max(1, len(working) // 3))
    coeffs = fit_yule_walker_ar(working, order)

    ar_next = ar_predict_next(working, coeffs)
    season = seasonal_baseline(original_series, config.seasonal_period)

    if config.seasonal_period > 0 and len(original_series) >= config.seasonal_period:
        combined = (ar_next + season) / 2.0
    else:
        combined = ar_next

    if prepared and config.diff_order > 0:
        forecast_value = invert_differences(original_series, combined, config.diff_order)
    else:
        forecast_value = combined

    observed = original_series[-1]
    residual = observed - forecast_value
    score, is_anomaly = detect_anomaly(observed, forecast_value, residual_history, config.anomaly_sigma)
    band_sd = stddev(residual_history) if len(residual_history) >= 5 else stddev(original_series)
    interval = max(0.0, config.anomaly_sigma * band_sd)

    return ForecastResult(
        value=forecast_value,
        lower=max(0.0, forecast_value - interval),
        upper=max(0.0, forecast_value + interval),
        residual=residual,
        anomaly_score=score,
        is_anomaly=is_anomaly,
    )


def update_model_state(state: InterfaceModelState, sample: PacketSample) -> None:
    state.samples.append(sample)
    state.total_series.append(sample.total_pps)
    state.rx_series.append(sample.rx_pps)
    state.tx_series.append(sample.tx_pps)

    state.ewma_total = update_ewma(state.ewma_total, sample.total_pps, state.config.ewma_alpha)
    state.ewma_rx = update_ewma(state.ewma_rx, sample.rx_pps, state.config.ewma_alpha)
    state.ewma_tx = update_ewma(state.ewma_tx, sample.tx_pps, state.config.ewma_alpha)

    total_values = list(state.total_series)
    residuals = list(state.residual_series)

    if len(total_values) >= state.config.warmup_samples:
        forecast = fit_forecast_for_series(total_values, state.config, residuals)
        state.last_forecast = forecast
        state.residual_series.append(forecast.residual)


def format_rate(value: float) -> str:
    units = ["pps", "Kpps", "Mpps", "Gpps"]
    idx = 0
    while value >= 1000.0 and idx < len(units) - 1:
        value /= 1000.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def compact_float(value: Optional[float], width: int = 10) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.2f}"


def render_table(states: List[InterfaceModelState]) -> None:
    print(
        f"{'IFACE':<12} {'RX':>12} {'TX':>12} {'TOTAL':>12} "
        f"{'EWMA':>12} {'FCST':>12} {'LOW':>12} {'HIGH':>12} "
        f"{'SCORE':>8} {'FLAG':>6}"
    )
    for state in states:
        latest = state.samples[-1] if state.samples else None
        forecast = state.last_forecast
        rx = format_rate(latest.rx_pps) if latest else "-"
        tx = format_rate(latest.tx_pps) if latest else "-"
        total = format_rate(latest.total_pps) if latest else "-"
        ewma = format_rate(state.ewma_total) if state.ewma_total is not None else "-"
        fcst = format_rate(forecast.value) if forecast else "-"
        low = format_rate(forecast.lower) if forecast else "-"
        high = format_rate(forecast.upper) if forecast else "-"
        score = f"{forecast.anomaly_score:.2f}" if forecast else "-"
        flag = "YES" if forecast and forecast.is_anomaly else "NO"
        print(
            f"{state.iface:<12} {rx:>12} {tx:>12} {total:>12} "
            f"{ewma:>12} {fcst:>12} {low:>12} {high:>12} "
            f"{score:>8} {flag:>6}"
        )


def render_json(states: List[InterfaceModelState]) -> None:
    payload = []
    for state in states:
        latest = state.samples[-1] if state.samples else None
        stats = compute_stats(list(state.total_series))
        payload.append(
            {
                "iface": state.iface,
                "latest": asdict(latest) if latest else None,
                "ewma_total": state.ewma_total,
                "ewma_rx": state.ewma_rx,
                "ewma_tx": state.ewma_tx,
                "forecast": asdict(state.last_forecast) if state.last_forecast else None,
                "stats": asdict(stats),
                "config": asdict(state.config),
            }
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


def render_details(states: List[InterfaceModelState]) -> None:
    for state in states:
        stats_total = compute_stats(list(state.total_series))
        stats_rx = compute_stats(list(state.rx_series))
        stats_tx = compute_stats(list(state.tx_series))
        print(f"\n[{state.iface}]")
        print(f"  samples         : {len(state.samples)}")
        print(f"  total_mean_pps  : {stats_total.mean:.2f}")
        print(f"  total_p95_pps   : {stats_total.p95:.2f}")
        print(f"  total_stdev_pps : {stats_total.stdev:.2f}")
        print(f"  total_trend     : {stats_total.trend:.4f}")
        print(f"  rx_mean_pps     : {stats_rx.mean:.2f}")
        print(f"  tx_mean_pps     : {stats_tx.mean:.2f}")
        print(f"  ewma_total      : {state.ewma_total if state.ewma_total is not None else 0.0:.2f}")
        if state.last_forecast:
            print(f"  forecast        : {state.last_forecast.value:.2f}")
            print(f"  lower           : {state.last_forecast.lower:.2f}")
            print(f"  upper           : {state.last_forecast.upper:.2f}")
            print(f"  residual        : {state.last_forecast.residual:.2f}")
            print(f"  anomaly_score   : {state.last_forecast.anomaly_score:.2f}")
            print(f"  anomaly         : {state.last_forecast.is_anomaly}")
        else:
            print("  forecast        : N/A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overengineered packet-count analyzer with AR-style forecasting")
    parser.add_argument("--iface", action="append", help="Interface to monitor, can be repeated")
    parser.add_argument("-i", "--interval", type=float, default=2.0, help="Sampling interval in seconds")
    parser.add_argument("-n", "--iterations", type=int, default=0, help="Number of iterations, 0 means infinite")
    parser.add_argument("--window", type=int, default=120, help="Rolling window size")
    parser.add_argument("--ar-order", type=int, default=5, help="Autoregressive order")
    parser.add_argument("--diff-order", type=int, default=0, help="Differencing order")
    parser.add_argument("--seasonal-period", type=int, default=0, help="Simple seasonal period")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup samples before forecasting")
    parser.add_argument("--sigma", type=float, default=3.0, help="Anomaly sigma threshold")
    parser.add_argument("--ewma-alpha", type=float, default=0.25, help="EWMA alpha")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--details", action="store_true", help="Emit detailed stats")
    return parser.parse_args()


def resolve_interfaces(requested: Optional[List[str]]) -> List[str]:
    data = read_proc_net_dev()
    available = sorted(data.keys())
    if not requested:
        return available
    valid = [iface for iface in requested if iface in data]
    if not valid:
        raise SystemExit("No valid interfaces selected")
    return valid


def build_states(ifaces: List[str], config: ModelConfig) -> List[InterfaceModelState]:
    return [InterfaceModelState(iface=iface, config=config) for iface in ifaces]


def initial_prime(states: List[InterfaceModelState]) -> None:
    for state in states:
        snap = sample_interface(state.iface)
        if snap is not None:
            state.snapshots.append(snap)


def collect_once(state: InterfaceModelState) -> None:
    snap = sample_interface(state.iface)
    if snap is None:
        return
    if state.snapshots:
        prev = state.snapshots[-1]
        sample = build_sample(prev, snap)
        if sample is not None:
            update_model_state(state, sample)
    state.snapshots.append(snap)


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()

    interval = max(args.interval, 0.2)
    config = ModelConfig(
        interval=interval,
        window_size=max(args.window, 10),
        ar_order=max(args.ar_order, 1),
        diff_order=max(args.diff_order, 0),
        seasonal_period=max(args.seasonal_period, 0),
        anomaly_sigma=max(args.sigma, 0.1),
        warmup_samples=max(args.warmup, 5),
        ewma_alpha=min(max(args.ewma_alpha, 0.01), 1.0),
    )

    ifaces = resolve_interfaces(args.iface)
    states = build_states(ifaces, config)
    initial_prime(states)

    count = 0
    while RUNNING:
        time.sleep(config.interval)

        for state in states:
            collect_once(state)

        if args.json:
            render_json(states)
        else:
            render_table(states)
            if args.details:
                render_details(states)
            print()

        count += 1
        if args.iterations > 0 and count >= args.iterations:
            break


if __name__ == "__main__":
    main()
