import concurrent.futures
import json
import os

class Searcher:
    @staticmethod
    def estimate_compute_time(dp, tp, pp, total_tflops, gpu_flops, overlap_ratio=0.6):
        effective_flops = dp * tp * pp * overlap_ratio
        tflops_per_gpu = total_tflops / effective_flops
        compute_time_sec = tflops_per_gpu / (gpu_flops / 1000)  # TFLOPs / TFLOPs/s
        return compute_time_sec * 1000  # ms

    @staticmethod
    def estimate_tp_time(tp_comm_sum_MB, bw_gpu_to_gpu):
        return (tp_comm_sum_MB / 1024) / bw_gpu_to_gpu * 1000  # ms

    @staticmethod
    def estimate_dp_time(dp_sum_gradient_MB, dp, n_gpu_per_node, bw_gpu_to_gpu, bw_inter_node):
        bandwidth = bw_gpu_to_gpu if dp <= n_gpu_per_node else bw_inter_node
        return (dp_sum_gradient_MB / 1024) / bandwidth * 1000  # ms

    @staticmethod
    def estimate_pp_time(layer_activation_MB, pp, bw_gpu_to_gpu):
        if pp <= 1:
            return 0.0
        pp_comm_MB = sum(layer_activation_MB[:pp - 1])
        return (pp_comm_MB / 1024) / bw_gpu_to_gpu * 1000  # ms

    @staticmethod
    def evaluate_config(args):
        dp, tp, pp, cluster_metadata, model_metadata = args

        # 하드웨어 정보
        n_node         = cluster_metadata["n_node"]
        n_gpu_per_node = cluster_metadata["n_gpu_per_node"]
        gpu_flops      = cluster_metadata["gpu_gflops"]
        gpu_memory     = cluster_metadata["gpu_memory"]     # unit : MB
        bw_gpu_to_gpu  = cluster_metadata["bw_gpu_to_gpu"]  # unit : GB/s
        bw_cpu_to_gpu  = cluster_metadata["bw_cpu_to_gpu"]  # unit : GB/s
        bw_cpu_to_cpu  = cluster_metadata["bw_cpu_to_cpu"]  # unit : GB/s
        bw_inter_node  = cluster_metadata["bw_inter_node"]  # unit : GB/s

        # 모델 정보
        global_batch_size   = model_metadata["global_batch_size"]
        tp_comm_sum_MB      = model_metadata["tp_comm_sum_MB"]
        tp_comm             = model_metadata["tp_comm"]
        dp_sum_gradient_MB  = model_metadata["dp_sum_gradient_MB"]
        num_layers          = model_metadata["num_layers"]
        layer_activation_MB = model_metadata["layer_activation_MB"]
        layer_tflops        = model_metadata["layer_tflops"]

        total_tflops = 3 * sum(layer_tflops)

        compute_time = Searcher.estimate_compute_time(dp, tp, pp, total_tflops, gpu_flops)
        tp_time = Searcher.estimate_tp_time(tp_comm_sum_MB, bw_gpu_to_gpu)
        dp_time = Searcher.estimate_dp_time(dp_sum_gradient_MB, dp, n_gpu_per_node, bw_gpu_to_gpu, bw_inter_node)
        pp_time = Searcher.estimate_pp_time(layer_activation_MB, pp, bw_gpu_to_gpu)

        total_time = compute_time + tp_time + dp_time + pp_time
        return (dp, tp, pp, total_time)

    @staticmethod
    def search(cluster_metadata: dict, model_metadata: dict, output_file: str = "output.json") -> tuple[int, int, int]:
        n_gpu = cluster_metadata["n_node"] * cluster_metadata["n_gpu_per_node"]
        num_layers = model_metadata["num_layers"]

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

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dp, tp, pp, score in executor.map(Searcher.evaluate_config, configs):
                config_key = f"{dp}, {tp}, {pp}"
                all_scores[config_key] = f"{score:.4f}ms/iter"
                if score < best_score:
                    best_score = score
                    best_config = (dp, tp, pp)

        # 결과 저장
        os.makedirs("./answers", exist_ok=True)
        output_path = os.path.join("./answers", output_file)
        with open(output_path, "w") as f:
            json.dump(all_scores, f, indent=4)

        return best_config if best_config else (1, 1, n_gpu)
