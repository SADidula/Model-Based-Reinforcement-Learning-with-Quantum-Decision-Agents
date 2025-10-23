import statistics
from typing import Dict, List, Optional, Tuple
import psutil
import os
import sys
import resource
import math
import os as _os  # local alias to read env without shadowing

class MetricsTracker:
    # Updated to per-step schema as requested
    CSV_FIELDS = [
        "Reward per step",
        "Efficiency (Reward per Step)",
        "Training/Test Time (s)",
        "Latency (s/step)",
        "Throughput (steps/s)",
        "Memory Consumed (MB)",
        "Stability (Reward Std)",
        "Convergence Rate per step",
    ]

    def __init__(self):
        self.total_steps = 0
        self.total_time_s = 0.0
        self.total_episodes = 0
        self.current_episode_reward = 0.0
        self._reset_window()

    def _reset_window(self):
        # We keep window buffers to support convergence-rate estimation and compatibility
        self.window_rewards: List[float] = []
        self.window_step_times_s: List[float] = []
        self.window_episode_returns: List[float] = []
        self.window_episodes = 0
        self.window_errors: List[float] = []
        self.window_error_times_s: List[float] = []
        self._window_time_acc_s: float = 0.0

    def record_step(self, reward: float, step_time_s: float, done: bool = False, error: Optional[float] = None):
        # Maintain history so convergence can be estimated across steps
        self.window_rewards.append(float(reward))
        self.window_step_times_s.append(float(step_time_s))
        self.total_steps += 1
        self.total_time_s += float(step_time_s)
        self.current_episode_reward += float(reward)

        self._window_time_acc_s += float(step_time_s)
        if error is not None:
            # Use absolute error for exponential envelope; epsilon to avoid zero
            e = float(abs(error))
            if e <= 0.0:
                e = 1e-12
            self.window_errors.append(e)
            self.window_error_times_s.append(self._window_time_acc_s)

        if done:
            self.window_episode_returns.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.window_episodes += 1
            self.total_episodes += 1

    def record_error(self, error: float, t_rel_s: Optional[float] = None):
        if t_rel_s is None:
            t_rel_s = self._window_time_acc_s
        e = float(abs(error))
        if e <= 0.0:
            e = 1e-12
        self.window_errors.append(e)
        self.window_error_times_s.append(float(t_rel_s))

    def _estimate_convergence_rate(self) -> float:
        # Estimate empirical exponential convergence rate r (1/s) from accumulated (time, error) samples
        times = self.window_error_times_s
        errs = self.window_errors
        debug = _os.environ.get("METRICS_DEBUG", "0") == "1"

        if len(times) < 3 or len(errs) < 3:
            if len(times) >= 2 and len(errs) >= 2:
                # Fallback two-point estimate
                valid: List[Tuple[float, float]] = []
                prev_t = -math.inf
                for t, e in zip(times, errs):
                    if e is None or not math.isfinite(e):
                        continue
                    if t is None or not math.isfinite(t):
                        continue
                    if t < prev_t:
                        prev_t = max(prev_t, t)
                        continue
                    valid.append((t, e if e > 0.0 else 1e-12))
                    prev_t = t
                if len(valid) >= 2:
                    t1, e1 = valid[0]
                    t2, e2 = valid[-1]
                    if e1 > 0 and e2 > 0 and t2 > t1:
                        r = -math.log(e2 / e1) / (t2 - t1)
                        r = float(max(0.0, r)) if math.isfinite(r) else 0.0
                        if debug:
                            print(f"[MetricsTracker] Two-point r={r:.6e}")
                        return r
            return 0.0

        valid: List[Tuple[float, float]] = []
        prev_t = -math.inf
        for t, e in zip(times, errs):
            if e is None or not math.isfinite(e):
                continue
            if t is None or not math.isfinite(t):
                continue
            if t < prev_t:
                prev_t = max(prev_t, t)
                continue
            valid.append((t, e if e > 0.0 else 1e-12))
            prev_t = t

        if len(valid) < 3:
            return 0.0

        xs = [t for t, _ in valid]
        ys = [math.log(max(e, 1e-12)) for _, e in valid]  # avoid log(0)

        n = float(len(xs))
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx <= 0.0 or not math.isfinite(sxx):
            return 0.0
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = sxy / sxx
        r = -slope
        if not math.isfinite(r) or r < 0.0:
            r = 0.0
        return float(r)

    def compute_step_metrics(self, last_reward: float, last_step_time_s: float) -> Dict[str, float]:
        # Per-step metrics based on the most recent step
        latency = float(last_step_time_s) if last_step_time_s > 0.0 else 0.0
        throughput = (1.0 / latency) if latency > 0.0 else 0.0
        mem_mb = self._get_process_memory_mb()
        # Stability (Std) for a single sample is 0.0
        stability_std = 0.0
        convergence_rate = self._estimate_convergence_rate()

        return {
            "Reward per step": float(last_reward),
            "Efficiency (Reward per Step)": float(last_reward),
            "Training/Test Time (s)": float(last_step_time_s),
            "Latency (s/step)": latency,
            "Throughput (steps/s)": throughput,
            "Memory Consumed (MB)": mem_mb,
            "Stability (Reward Std)": stability_std,
            "Convergence Rate per step": convergence_rate,
        }

    # The window-based method remains for compatibility; unused in per-step mode
    def compute_window_metrics(self) -> Dict[str, float]:
        n = len(self.window_rewards)
        total_reward = float(sum(self.window_rewards))
        avg_reward = float(sum(self.window_episode_returns)) / float(len(self.window_episode_returns)) if self.window_episode_returns else (total_reward / n if n > 0 else 0.0)
        efficiency = (total_reward / n) if n > 0 else 0.0
        window_time_s = float(sum(self.window_step_times_s))
        latency = (window_time_s / n) if n > 0 else 0.0
        throughput = (n / window_time_s) if window_time_s > 0 else 0.0
        mem_mb = self._get_process_memory_mb()
        stability = statistics.stdev(self.window_rewards) if n > 1 else 0.0
        convergence_rate_episodes = float(self.window_episodes)
        empirical_r = self._estimate_convergence_rate()

        # Not used for per-step CSV, but returned for potential snapshots
        return {
            "Reward per step": 0.0,  # placeholder to keep keys coherent if ever written accidentally
            "Efficiency (Reward per Step)": efficiency,
            "Training/Test Time (s)": window_time_s,
            "Latency (s/step)": latency,
            "Throughput (steps/s)": throughput,
            "Memory Consumed (MB)": mem_mb,
            "Stability (Reward Std)": stability,
            "Convergence Rate per step": empirical_r
        }

    def flush_window(self) -> Dict[str, float]:
        metrics = self.compute_window_metrics()
        self._reset_window()
        return metrics

    def _get_process_memory_mb(self) -> float:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == 'darwin':
                    return usage.ru_maxrss / (1024.0 * 1024.0)
                else:
                    return usage.ru_maxrss / 1024.0
            except Exception:
                return float('nan')