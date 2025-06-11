import concurrent.futures
import json
import os

# Constants for performance modeling
BASE_EFF                = 0.35   # Base single-GPU efficiency
DP_EFF_PENALTY_FACTOR   = 0.20   # Efficiency drop per additional DP unit
TP_EFF_PENALTY_FACTOR   = 0.25   # Efficiency drop per additional TP unit
OVERLAP_RATIO_PP        = 0.60   # Overlap ratio for 1F1B pipeline schedule

class Searcher:

    @staticmethod
    def estimate_compute_time(dp, tp, pp, total_tflops, gpu_flops):
        eff = BASE_EFF
        eff /= (1 + DP_EFF_PENALTY_FACTOR * (dp - 1))
        eff /= (1 + TP_EFF_PENALTY_FACTOR * (tp - 1))
        eff /= (1 + 0.1 * abs(dp - tp))                     # Penalty for imbalance between DP and TP

        perf = gpu_flops / 1000 * dp * tp                   # Total TFLOPs/s across GPUs
        compute_time_sec = total_tflops / perf / eff

        if pp > 1:
            compute_time_sec /= OVERLAP_RATIO_PP            # Apply pipeline overlap

        return compute_time_sec * 1000                      # Convert to ms

    @staticmethod
    def estimate_dp_time(dp_sum_gradient_MB, dp, n_gpu_per_node, bw_gpu_to_gpu, bw_inter_node):
        if dp == 1:
            return 0.0
        comm_MB = dp_sum_gradient_MB * 2 * (dp - 1) / dp
        bandwidth = bw_gpu_to_gpu if dp <= n_gpu_per_node else bw_inter_node
        bw_MBps = bandwidth * 1024                          # Convert GB/s to MB/s
        return (comm_MB / bw_MBps) * 1000                   # Convert to ms

    @staticmethod
    def estimate_tp_time(tp_comm_sum_MB, tp, bw_gpu_to_gpu):
        if tp == 1:
            return 0.0
        comm_MB = (tp_comm_sum_MB / tp) * 2 * (tp - 1) / tp
        bw_MBps = bw_gpu_to_gpu * 1024                      # Convert GB/s to MB/s
        return (comm_MB / bw_MBps) * 1000                   # Convert to ms

    @staticmethod
    def estimate_pp_time(layer_activation_MB, pp, bw_gpu_to_gpu):
        if pp == 1:
            return 0.0
        act_MB = max(layer_activation_MB)
        comm_MB = (pp - 1) * 2 * act_MB
        bw_MBps = bw_gpu_to_gpu * 1024                      # Convert GB/s to MB/s
        return (comm_MB / bw_MBps) * 1000                   # Convert to ms

    @staticmethod
    def evaluate_config(args):
        dp, tp, pp, cluster_metadata, model_metadata = args

        # Cluster specifications
        n_gpu_per_node = cluster_metadata["n_gpu_per_node"]
        gpu_flops      = cluster_metadata["gpu_gflops"]
        bw_gpu_to_gpu  = cluster_metadata["bw_gpu_to_gpu"]  # GB/s
        bw_inter_node  = cluster_metadata["bw_inter_node"]  # GB/s

        # Model metadata
        tp_comm_sum_MB      = model_metadata["tp_comm_sum_MB"]
        dp_sum_gradient_MB  = model_metadata["dp_sum_gradient_MB"]
        layer_activation_MB = model_metadata["layer_activation_MB"]
        layer_tflops        = model_metadata["layer_tflops"]

        total_tflops = 3 * sum(layer_tflops)                # Forward + Backward FLOPs

        # Estimate each component of the iteration time
        compute_time = Searcher.estimate_compute_time(dp, tp, pp, total_tflops, gpu_flops)
        dp_time = Searcher.estimate_dp_time(dp_sum_gradient_MB, dp, n_gpu_per_node, bw_gpu_to_gpu, bw_inter_node)
        tp_time = Searcher.estimate_tp_time(tp_comm_sum_MB, tp, bw_gpu_to_gpu)
        pp_time = Searcher.estimate_pp_time(layer_activation_MB, pp, bw_gpu_to_gpu)

        total_time = compute_time + tp_time + dp_time + pp_time
        return (dp, tp, pp, total_time)

    @staticmethod
    def search(cluster_metadata: dict, model_metadata: dict, output_file: str = "output.json") -> tuple[int, int, int]:
        n_gpu = cluster_metadata["n_node"] * cluster_metadata["n_gpu_per_node"]
        num_layers = model_metadata["num_layers"]

        # Generate all valid (dp, tp, pp) combinations where dp * tp * pp == n_gpu
        configs = [
            (dp, tp, pp, cluster_metadata, model_metadata)
            for dp in range(1, n_gpu + 1)
            for tp in range(1, n_gpu + 1)
            for pp in range(1, num_layers + 1)
            if dp * tp * pp == n_gpu
        ]

        best_config = None
        best_score = float('inf')
        all_scores = {}

        # Parallel evaluation
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dp, tp, pp, score in executor.map(Searcher.evaluate_config, configs):
                config_key = f"{dp}, {tp}, {pp}"
                all_scores[config_key] = f"{score:.4f}ms/iter"
                if score < best_score:
                    best_score = score
                    best_config = (dp, tp, pp)

        # Save all results to file
        os.makedirs("./answers", exist_ok=True)
        output_path = os.path.join("./answers", output_file)
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=4)

        return best_config if best_config else (1, 1, n_gpu)
