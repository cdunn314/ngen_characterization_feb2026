"""
adjust_energy_groups.py

Try alternative energy-group structures, rebin the response matrix accordingly,
call solve_flux_spectrum for each, and report condition numbers and mean relative errors.
Pick the grouping with lowest mean relative error subject to reasonable conditioning.
"""

import numpy as np
from process_foil_data import solve_flux_spectrum, build_response_matrix_from_processed_data

def rebin_response_matrix(response_matrix, orig_edges, new_edges):
    # orig_edges: length M+1 for M original groups (eV)
    # new_edges: length N+1 for N new groups (eV)
    orig_mids = 0.5 * (orig_edges[:-1] + orig_edges[1:])
    M = response_matrix.shape[1]
    if len(orig_edges) - 1 != M:
        raise ValueError("orig_edges length does not match response_matrix columns")
    new_cols = []
    for i in range(len(new_edges) - 1):
        mask = (orig_mids >= new_edges[i]) & (orig_mids < new_edges[i + 1])
        if not np.any(mask):
            # If no original bin falls inside new bin, pick the closest original bin
            idx = np.argmin(np.abs(orig_mids - 0.5 * (new_edges[i] + new_edges[i + 1])))
            col = response_matrix[:, idx:idx+1].copy()
        else:
            col = np.sum(response_matrix[:, mask], axis=1, keepdims=True)
        new_cols.append(col)
    return np.hstack(new_cols)

def evaluate_groupings(response_matrix, responses, response_errs, orig_edges, candidate_edges_list):
    results = []
    for edges in candidate_edges_list:
        try:
            R_rebinned = rebin_response_matrix(response_matrix, orig_edges, edges)
        except Exception as e:
            results.append((edges, None, None, np.inf, str(e)))
            continue
        try:
            phi, phi_err, residuals = solve_flux_spectrum(responses, response_errs, R_rebinned, edges)
        except Exception as e:
            results.append((edges, None, None, np.linalg.cond(R_rebinned), f"solve error: {e}"))
            continue
        # stability metrics
        cond = np.linalg.cond(R_rebinned)
        # avoid division by zero in relative error calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.abs(phi_err / np.where(np.abs(phi) > 0, np.abs(phi), np.nan))
            rel_err = rel_err[~np.isnan(rel_err)]
        mean_rel_err = float(np.nan) if rel_err.size == 0 else float(np.mean(rel_err))
        results.append((edges, phi, phi_err, cond, mean_rel_err))
    return results

if __name__ == "__main__":
    # load processed data (same call pattern used in the notebook)
    response_matrix, responses, response_errs = build_response_matrix_from_processed_data(angle=90)
    responses = np.asarray(responses)
    response_errs = np.asarray(response_errs)

    # Original group edges used in the notebook (eV)
    orig_edges = np.array([0, 1, 4, 8, 12, 16]) * 1e6
    if response_matrix.shape[1] != len(orig_edges) - 1:
        # fallback: assume equal-width groups spanning 0..15e6
        orig_edges = np.linspace(0, 15e6, response_matrix.shape[1] + 1)

    # Candidate re-binnings to try (examples)
    candidates = [
        np.array([0, 3.0e6, 12.0e6, 16.0e6]),                 # 3 bins (coarse)
        np.array([0, 1.0e6, 6.0e6, 12.0e6, 16.0e6]),         # 4 bins
        np.array([0, 1.0e6, 4.0e6, 8.0e6, 12.0e6, 16.0e6]),  # 5 bins (slightly coarser)
        np.array([0, 1.0e6, 3.0e6, 8.0e6, 16.0e6]),          # hybrid
    ]

    results = evaluate_groupings(response_matrix, responses, response_errs, orig_edges, candidates)

    # Print concise summary and recommend grouping with lowest mean relative error while keeping cond reasonable
    best = None
    for edges, phi, phi_err, cond, metric in results:
        if phi is None:
            print(f"Edges {edges/1e6} MeV -> failed (cond={cond})")
            continue
        print(f"Edges {edges/1e6} MeV -> cond={cond:.2e}, mean_rel_err={metric:.2%}")
        if np.isfinite(metric):
            if best is None or (metric < best[4] and cond < 1e6):
                best = (edges, phi, phi_err, cond, metric)

    if best is not None:
        edges, phi_best, phi_err_best, cond_best, metric_best = best
        print("\nRecommended grouping:", (edges / 1e6).tolist(), "MeV")
        print(f"  Condition number: {cond_best:.2e}")
        print(f"  Mean relative error: {metric_best:.2%}")
    else:
        print("\nNo satisfactory re-binning found; consider stronger regularization or reducing detector weighting.")