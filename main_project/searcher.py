import concurrent.futures
import json
import os

BASE_EFF                = 0.35   # 단일-GPU 연산 효율
DP_EFF_PENALTY_FACTOR   = 0.20   # Data-parallel 증가 시 효율 저하 계수
TP_EFF_PENALTY_FACTOR   = 0.25   # Tensor-parallel 증가 시 효율 저하 계수
OVERLAP_RATIO_PP        = 0.60   # 1F1B 파이프라인 겹침 비율

class Searcher:

    @staticmethod
    def estimate_compute_time(dp, tp, pp, total_tflops, gpu_flops):
        eff = BASE_EFF
        eff /= (1 + DP_EFF_PENALTY_FACTOR * (dp - 1))
        eff /= (1 + TP_EFF_PENALTY_FACTOR * (tp - 1))
        eff /= (1 + 0.1 * abs(dp - tp))          # 불균형 벌점
        
        compute_time_sec = total_tflops / (gpu_flops / 1000 * dp * tp) / eff # TFLOPs / TFLOPs/s
        
        if pp > 1:
            compute_time_sec /= OVERLAP_RATIO_PP
        return compute_time_sec * 1000  # ms

    @staticmethod
    def estimate_dp_time(dp_sum_gradient_MB, dp, n_gpu_per_node, bw_gpu_to_gpu, bw_inter_node):
        if dp == 1:
            return 0.0
        comm_MB = dp_sum_gradient_MB * 2 * (dp - 1) / dp
        bandwidth = bw_gpu_to_gpu if dp <= n_gpu_per_node else bw_inter_node
        bw_MBps = bandwidth * 1024  # GB/s -> MB/s
        return (comm_MB / bw_MBps) * 1000  # ms

    
    @staticmethod
    def estimate_tp_time(tp_comm_sum_MB, tp, bw_gpu_to_gpu):
        if tp == 1:
            return 0.0
        comm_MB = (tp_comm_sum_MB / tp) * 2 * (tp - 1) / tp
        bw_MBps = bw_gpu_to_gpu * 1024
        return (comm_MB / bw_MBps) * 1000  # ms


    @staticmethod
    def estimate_pp_time(layer_activation_MB, pp, bw_gpu_to_gpu):
        if pp == 1:
            return 0.0
        act_MB = max(layer_activation_MB)
        comm_MB = (pp - 1) * 2 * act_MB
        bw_MBps = bw_gpu_to_gpu * 1024            # GB/s → MB/s
        return (comm_MB / bw_MBps) * 1000         # ms

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
        dp_time = Searcher.estimate_dp_time(dp_sum_gradient_MB, dp, n_gpu_per_node, bw_gpu_to_gpu, bw_inter_node)
        tp_time = Searcher.estimate_tp_time(tp_comm_sum_MB, tp, bw_gpu_to_gpu)
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
