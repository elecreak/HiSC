import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import csgraph
MSEM = 0.95
CDST = 28.0
CSMO = 56.0
PDEC = 0.4
HISC_RATIO = 0.9
PRUNE_LAYERS = [1, 4, 7, 16, 23]

class HiSCState:

    def __init__(self, coords, special_indices, original_indices, clusters):
        self.coords = coords
        self.special_indices = special_indices
        self.original_indices = original_indices
        self.clusters = clusters
        self.gamma_base = None
        self.target_ratio = None

def cluster_tokens(image_features_ToMe, coords_ToMe, num_semantic_clusters=16):
    if image_features_ToMe.dim() == 3:
        (B, N_patch, D) = image_features_ToMe.shape
        N = B * N_patch
    else:
        (N, D) = image_features_ToMe.shape
    device = image_features_ToMe.device
    features_f32 = image_features_ToMe.reshape(N, D).float()
    coords_f32 = coords_ToMe.view(N, 3).float()
    min_xyz = coords_f32.min(dim=0)[0]
    max_xyz = coords_f32.max(dim=0)[0]
    scene_max_span = torch.norm(max_xyz - min_xyz, p=2)
    dist_mat_global = torch.cdist(coords_f32, coords_f32, p=2)
    sigma = scene_max_span / CSMO
    weights = torch.exp(-dist_mat_global ** 2 / (2 * sigma ** 2 + 1e-06))
    weights_norm = weights / weights.sum(dim=1, keepdim=True)
    diffused_features = torch.matmul(weights_norm, features_f32)
    features_norm = torch.nn.functional.normalize(diffused_features, dim=-1)
    centers = features_norm[torch.linspace(0, N - 1, num_semantic_clusters, dtype=torch.long, device=device)]
    labels_sem = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(5):
        dists = torch.cdist(features_norm, centers, p=2)
        labels_sem = torch.argmin(dists, dim=1)
        H = torch.nn.functional.one_hot(labels_sem, num_classes=num_semantic_clusters).float()
        cluster_sizes_tensor = H.sum(dim=0).clamp(min=1e-06)
        new_centers = torch.matmul(H.T, features_norm) / cluster_sizes_tensor.unsqueeze(1)
        centers = torch.nn.functional.normalize(new_centers, dim=-1)
    tau_dist = scene_max_span / CDST
    final_clusters = []
    final_instance_labels = torch.zeros(N, dtype=torch.long, device=device)
    current_instance_id = 0
    for k in range(num_semantic_clusters):
        idx_k = torch.nonzero(labels_sem == k).squeeze(-1)
        if len(idx_k) == 0:
            continue
        if len(idx_k) == 1:
            final_clusters.append(idx_k)
            final_instance_labels[idx_k] = current_instance_id
            current_instance_id += 1
            continue
        dist_mat_k = dist_mat_global[idx_k][:, idx_k]
        adj_k = dist_mat_k <= tau_dist
        adj_np = adj_k.cpu().numpy()
        graph = sparse.csr_matrix(adj_np)
        (n_comps, sub_labels) = csgraph.connected_components(csgraph=graph, directed=False)
        sub_labels_tensor = torch.tensor(sub_labels, device=device)
        for sub_id in range(n_comps):
            sub_idx = torch.nonzero(sub_labels_tensor == sub_id).squeeze(-1)
            global_sub_idx = idx_k[sub_idx]
            final_clusters.append(global_sub_idx)
            final_instance_labels[global_sub_idx] = current_instance_id
            current_instance_id += 1
    num_keep = 32
    if current_instance_id > num_keep:
        bincounts = torch.bincount(final_instance_labels)
        (_, topk_indices) = torch.topk(bincounts, num_keep)
        keep_mask = torch.isin(final_instance_labels, topk_indices)
        orphan_mask = ~keep_mask
        if orphan_mask.any():
            anchor_indices = torch.nonzero(keep_mask).squeeze(-1)
            orphan_indices = torch.nonzero(orphan_mask).squeeze(-1)
            dist_orphan_to_anchor = dist_mat_global[orphan_indices][:, anchor_indices]
            nearest_anchor_local_idx = torch.argmin(dist_orphan_to_anchor, dim=1)
            nearest_anchor_global_idx = anchor_indices[nearest_anchor_local_idx]
            final_instance_labels[orphan_indices] = final_instance_labels[nearest_anchor_global_idx]
        unique_labels = torch.unique(final_instance_labels)
        final_clusters = []
        new_labels = torch.zeros_like(final_instance_labels)
        for (new_id, old_id) in enumerate(unique_labels):
            idx = torch.nonzero(final_instance_labels == old_id).squeeze(-1)
            final_clusters.append(idx)
            new_labels[idx] = new_id
        final_instance_labels = new_labels
    torch.manual_seed(24)
    torch.rand(len(final_clusters), 3, device=device)
    return final_clusters

def merge_tokens(image_features_ToMe, coords_ToMe):
    (B, N_patch, D) = image_features_ToMe.shape
    N = B * N_patch
    orig_feat_dtype = image_features_ToMe.dtype
    orig_coord_dtype = coords_ToMe.dtype
    device = image_features_ToMe.device
    features = image_features_ToMe.reshape(N, D)
    coords = coords_ToMe.view(N, 3)
    coords_f32 = coords.float()
    dist_matrix = torch.cdist(coords_f32, coords_f32, p=2)
    max_distance = dist_matrix.max()
    tau_dist = max_distance * 2.0 / 14.0
    D_adj = dist_matrix <= tau_dist
    features_f32 = features.float()
    features_norm = torch.nn.functional.normalize(features_f32, dim=-1)
    sim_matrix = torch.matmul(features_norm, features_norm.T)
    eye_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim_matrix_no_diag = sim_matrix.masked_fill(eye_mask, -1.0)
    max_sims_per_token = sim_matrix_no_diag.max(dim=1)[0]
    M_sim = max_sims_per_token.max()
    m_sim = max_sims_per_token.min()
    tau_sem = m_sim + MSEM * (M_sim - m_sim)
    S_adj = sim_matrix >= tau_sem
    A_adj = S_adj & D_adj
    A_adj_np = A_adj.cpu().numpy()
    graph = sparse.csr_matrix(A_adj_np)
    (n_components, labels) = csgraph.connected_components(csgraph=graph, directed=False)
    labels_tensor = torch.tensor(labels, device=device)
    H = torch.nn.functional.one_hot(labels_tensor.to(torch.int64), num_classes=n_components).float()
    cluster_sizes = H.sum(dim=0)
    H_norm = H / cluster_sizes.unsqueeze(0)
    merged_features = torch.matmul(H_norm.T, features_f32)
    merged_coords = torch.matmul(H_norm.T, coords_f32)
    weights = torch.arange(N, 0, -1, device=device, dtype=torch.float32).unsqueeze(1)
    anchor_indices = (H * weights).argmax(dim=0)
    updated_features = features.clone()
    updated_coords = coords.clone()
    updated_features[anchor_indices] = merged_features.to(orig_feat_dtype)
    updated_coords[anchor_indices] = merged_coords.to(orig_coord_dtype)
    keep_indices_tensor = torch.sort(anchor_indices)[0]
    updated_features_3d = updated_features.view(B, N_patch, D)
    updated_coords_3d = updated_coords.view(B, N_patch, 3)
    return (updated_features_3d, updated_coords_3d, keep_indices_tensor)

def select_tokens(state, prune_layers: list, layer_idx: int, image_features: torch.Tensor, coords_for_prun: torch.Tensor, mean_attn: torch.Tensor, max_sim_to_text: torch.Tensor, clusters: list, avg_prune_ratio: float, special_indices: torch.Tensor=None) -> tuple:
    N_curr = image_features.shape[0]
    device = image_features.device
    num_clusters = len(clusters)
    if special_indices is not None:
        normal_mask = torch.ones(N_curr, dtype=torch.bool, device=device)
        normal_mask[special_indices] = False
        normal_indices = torch.nonzero(normal_mask).squeeze(-1)
        clusters = [normal_indices[c] for c in clusters] + [special_indices]
        num_clusters = len(clusters)
        full_coords = torch.zeros((N_curr, 3), device=device)
        full_coords[normal_indices] = coords_for_prun
        coords_for_prun = full_coords
    if layer_idx not in prune_layers:
        if special_indices is not None:
            return (torch.arange(N_curr - len(special_indices), device=device), None, special_indices)
        return (torch.arange(N_curr, device=device), None)
    decay_factor = PDEC
    if state.gamma_base is None or state.target_ratio != avg_prune_ratio:
        target_sum = 28.0 * (1.0 - avg_prune_ratio)
        (low, high) = (0.0, 1.0)
        stage_durations = [prune_layers[i + 1] - prune_layers[i] for i in range(len(prune_layers) - 1)]
        stage_durations.append(28 - prune_layers[-1])
        for _ in range(40):
            mid = (low + high) / 2.0
            current_sum = 1.0
            cumulative = 1.0
            for k in range(len(prune_layers)):
                r_k = 1.0 - (1.0 - mid) * decay_factor ** k
                cumulative *= r_k
                current_sum += cumulative * stage_durations[k]
            if current_sum < target_sum:
                low = mid
            else:
                high = mid
        state.gamma_base = (low + high) / 2.0
        state.target_ratio = avg_prune_ratio
    prune_step_idx = prune_layers.index(layer_idx)
    current_ratio = 1.0 - (1.0 - state.gamma_base) * decay_factor ** prune_step_idx
    N_keep = max(1, int(N_curr * current_ratio))
    N_keep = max(N_keep, num_clusters)
    if N_keep >= N_curr:
        if special_indices is not None:
            return (torch.arange(N_curr - len(special_indices), device=device), None, special_indices)
        return (torch.arange(N_curr, device=device), None)
    (sim_min, sim_max) = (max_sim_to_text.min(), max_sim_to_text.max())
    sim_norm = (max_sim_to_text - sim_min) / (sim_max - sim_min + 1e-06)
    if mean_attn is not None:
        alpha_imp = max(0.0, min(1.0, (layer_idx - 1) / 26.0))
        t = alpha_imp ** 3
        (attn_min, attn_max) = (mean_attn.min(), mean_attn.max())
        attn_norm = (mean_attn - attn_min) / (attn_max - attn_min + 1e-06)
    else:
        attn_norm = torch.zeros_like(sim_norm)
        t = 0.0
    token_importance = (1.0 - t) * sim_norm + t * attn_norm
    cluster_importances = torch.zeros(num_clusters, device=device)
    cluster_sizes = torch.zeros(num_clusters, dtype=torch.long, device=device)
    for (i, c_indices) in enumerate(clusters):
        cluster_importances[i] = token_importance[c_indices].mean()
        cluster_sizes[i] = len(c_indices)
    if special_indices is not None and num_clusters > 1:
        cluster_importances[-1] = cluster_importances[:-1].min()
    (ci_min, ci_max) = (cluster_importances.min(), cluster_importances.max())
    cluster_importances = (cluster_importances - ci_min) / (ci_max - ci_min + 1e-08)
    if layer_idx <= 29:
        quotas = torch.ones(num_clusters, dtype=torch.long, device=device)
    else:
        quotas = torch.zeros(num_clusters, dtype=torch.long, device=device)
    quotas = torch.minimum(quotas, cluster_sizes)
    N_remain = N_keep - quotas.sum().item()
    if N_remain > 0:
        sharpened_imp = cluster_importances ** 2
        while N_remain > 0:
            unfilled_mask = quotas < cluster_sizes
            if not unfilled_mask.any():
                break
            active_imp = sharpened_imp * unfilled_mask.float()
            if active_imp.sum() == 0:
                active_imp = unfilled_mask.float()
            imp_sum = active_imp.sum() + 1e-06
            step_quotas = (N_remain * (active_imp / imp_sum)).long()
            if step_quotas.sum() == 0:
                (_, max_idx) = active_imp.max(dim=0)
                step_quotas[max_idx] = 1
            for i in range(num_clusters):
                if step_quotas[i] > 0:
                    allowed = cluster_sizes[i] - quotas[i]
                    actual_add = min(step_quotas[i].item(), allowed.item(), N_remain)
                    quotas[i] += actual_add
                    N_remain -= actual_add
                    if N_remain <= 0:
                        break
    alpha_intra = (layer_idx - 1) / 26.0
    alpha_intra = max(0.0, min(1.0, alpha_intra))
    ratio_rel = 0.95 * alpha_intra
    ratio_rel = 1.0 - (1.0 - ratio_rel) ** 3
    keep_indices_list = []
    for (i, c_indices) in enumerate(clusters):
        quota_c = quotas[i].item()
        N_c = len(c_indices)
        if quota_c >= N_c:
            keep_indices_list.append(c_indices)
            continue
        n_rel = int(quota_c * ratio_rel)
        if quota_c == 1:
            (n_rel, n_div) = (1, 0)
        else:
            n_div = quota_c - n_rel
        c_imp = token_importance[c_indices]
        c_selected_mask = torch.zeros(N_c, dtype=torch.bool, device=device)
        if n_rel > 0:
            (_, top_rel_idx_local) = torch.topk(c_imp, n_rel)
            c_selected_mask[top_rel_idx_local] = True
        if n_div > 0:
            c_coords = coords_for_prun[c_indices].float()
            c_feats = image_features[c_indices].float()
            d3d = torch.cdist(c_coords, c_coords)
            d3d_norm = d3d / (d3d.max() + 1e-06)
            c_feats_norm = torch.nn.functional.normalize(c_feats, dim=-1)
            dsem_norm = (1.0 - torch.matmul(c_feats_norm, c_feats_norm.T)) / 2.0
            d_comb = d3d_norm + dsem_norm
            if n_rel > 0:
                min_dists = d_comb[:, top_rel_idx_local].min(dim=1)[0]
            else:
                c_center_coord = c_coords.mean(dim=0, keepdim=True)
                c_center_feat = torch.nn.functional.normalize(c_feats.mean(dim=0, keepdim=True), dim=-1)
                d3d_c = torch.cdist(c_coords, c_center_coord).squeeze(-1) / (d3d.max() + 1e-06)
                dsem_c = (1.0 - torch.matmul(c_feats_norm, c_center_feat.T).squeeze(-1)) / 2.0
                min_dists = d3d_c + dsem_c
            min_dists.masked_fill_(c_selected_mask, 0.0)
            probs = min_dists + 1e-05
            probs = probs * (~c_selected_mask).float()
            actual_n_div = min(n_div, (~c_selected_mask).sum().item())
            if actual_n_div > 0:
                sampled_local_idx = torch.multinomial(probs, actual_n_div, replacement=False)
                c_selected_mask[sampled_local_idx] = True
        selected_global_idx = c_indices[c_selected_mask]
        keep_indices_list.append(selected_global_idx)
    final_keep_indices = torch.cat(keep_indices_list)
    final_keep_indices = torch.sort(final_keep_indices)[0]
    if special_indices is not None:
        kept_is_special = torch.isin(final_keep_indices, special_indices)
        kept_full_normal = final_keep_indices[~kept_is_special]
        kept_special = final_keep_indices[kept_is_special]
        inverse_normal = torch.full((N_curr,), -1, dtype=torch.long, device=device)
        inverse_normal[normal_indices] = torch.arange(len(normal_indices), device=device)
        attn_indices = inverse_normal[kept_full_normal]
        return (attn_indices, None, kept_special)
    return (final_keep_indices, None)

def detrend_attention(mean_attn, target_window_size=15):
    seq_len = mean_attn.shape[1]
    w = min(target_window_size, seq_len)
    if w % 2 == 0:
        w -= 1
    if w >= 3:
        padding = w // 2
        x = mean_attn.unsqueeze(1)
        x_padded = F.pad(x, (padding, padding), mode='reflect')
        baseline = F.avg_pool1d(x_padded, kernel_size=w, stride=1).squeeze(1)
        eps = 1e-06
        detrended_attn = mean_attn / torch.clamp(baseline, min=eps)
        return torch.nan_to_num(detrended_attn, nan=1.0, posinf=1.0)
    return mean_attn
