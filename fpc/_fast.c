/*
 * GenoCortex native C acceleration kernel (fpc._fast)
 * ===================================================
 *
 * Implements IEEE-754 numerically equivalent hot loops for:
 *   1. Chaos Game Representation trajectory embedding (cgr_embed)
 *   2. Exact spatial-grid accelerated online cortex learning (cortex_stream)
 *   3. Exact spatial-grid accelerated frozen batch scoring (cortex_score_batch)
 *
 * Uses CPython's PEP 3118 buffer protocol (PyObject_GetBuffer) exclusively --
 * zero compile-time dependency on NumPy C headers and zero ABI breakage across
 * NumPy 1.x / 2.x releases.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Adaptive power-of-2 spatial grid (32, 128, or 256 per axis): multiplying or
 * dividing by 2^p incurs zero IEEE-754 mantissa rounding error when computing
 * exact cell bounding intervals, while keeping cell occupancy O(1) from 100
 * prototypes up to 1,000,000+ megabase prototypes. */
static inline int choose_grid_n(int cap) {
    if (cap <= 8192) return 32;
    if (cap <= 131072) return 128;
    return 256;
}

static inline int cell_coord(double u, int grid_n) {
    int g = (int)(u * (double)grid_n);
    if (g < 0) return 0;
    if (g >= grid_n) return grid_n - 1;
    return g;
}

static inline double cell_axis_dist2(double u, int g, int grid_n, double inv_g) {
    if (g > 0) {
        double lo = (double)g * inv_g;
        if (u < lo) {
            double d = lo - u;
            return d * d;
        }
    }
    if (g < grid_n - 1) {
        double hi = (double)(g + 1) * inv_g;
        if (u > hi) {
            double d = u - hi;
            return d * d;
        }
    }
    return 0.0;
}

typedef struct {
    int grid_n;
    int grid_cells;
    double inv_g;
    int *head;
    int *next;
    int *prev;
    int *cell_of;
    int cap;
} SpatialGrid;

static int grid_init(SpatialGrid *grid, int cap, const double *X, int m) {
    int gn = choose_grid_n(cap > m ? cap : m);
    grid->grid_n = gn;
    grid->grid_cells = gn * gn;
    grid->inv_g = 1.0 / (double)gn;
    grid->cap = cap;
    grid->head = (int *)malloc((size_t)grid->grid_cells * sizeof(int));
    if (!grid->head) {
        grid->next = grid->prev = grid->cell_of = NULL;
        return -1;
    }
    for (int c = 0; c < grid->grid_cells; ++c) {
        grid->head[c] = -1;
    }
    if (cap <= 0) {
        grid->next = NULL;
        grid->prev = NULL;
        grid->cell_of = NULL;
        return 0;
    }
    grid->next = (int *)malloc((size_t)cap * sizeof(int));
    grid->prev = (int *)malloc((size_t)cap * sizeof(int));
    grid->cell_of = (int *)malloc((size_t)cap * sizeof(int));
    if (!grid->next || !grid->prev || !grid->cell_of) {
        free(grid->head);
        free(grid->next);
        free(grid->prev);
        free(grid->cell_of);
        grid->head = grid->next = grid->prev = grid->cell_of = NULL;
        return -1;
    }
    for (int i = 0; i < m; ++i) {
        int gx = cell_coord(X[2 * i], gn);
        int gy = cell_coord(X[2 * i + 1], gn);
        int cid = gy * gn + gx;
        grid->cell_of[i] = cid;
        grid->prev[i] = -1;
        grid->next[i] = grid->head[cid];
        if (grid->head[cid] != -1) {
            grid->prev[grid->head[cid]] = i;
        }
        grid->head[cid] = i;
    }
    return 0;
}

static void grid_free(SpatialGrid *grid) {
    free(grid->head);
    free(grid->next);
    free(grid->prev);
    free(grid->cell_of);
    grid->head = grid->next = grid->prev = grid->cell_of = NULL;
}

static inline void grid_insert(SpatialGrid *grid, int idx, double x, double y) {
    int gn = grid->grid_n;
    int gx = cell_coord(x, gn);
    int gy = cell_coord(y, gn);
    int cid = gy * gn + gx;
    grid->cell_of[idx] = cid;
    grid->prev[idx] = -1;
    grid->next[idx] = grid->head[cid];
    if (grid->head[cid] != -1) {
        grid->prev[grid->head[cid]] = idx;
    }
    grid->head[cid] = idx;
}

static inline void grid_move(SpatialGrid *grid, int idx, double x, double y) {
    int gn = grid->grid_n;
    int gx = cell_coord(x, gn);
    int gy = cell_coord(y, gn);
    int new_cid = gy * gn + gx;
    int old_cid = grid->cell_of[idx];
    if (new_cid == old_cid) return;

    int p = grid->prev[idx];
    int n = grid->next[idx];
    if (p != -1) {
        grid->next[p] = n;
    } else {
        grid->head[old_cid] = n;
    }
    if (n != -1) {
        grid->prev[n] = p;
    }

    grid->cell_of[idx] = new_cid;
    grid->prev[idx] = -1;
    grid->next[idx] = grid->head[new_cid];
    if (grid->head[new_cid] != -1) {
        grid->prev[grid->head[new_cid]] = idx;
    }
    grid->head[new_cid] = idx;
}

static void grid_rebuild(SpatialGrid *grid, const double *X, int m) {
    for (int c = 0; c < grid->grid_cells; ++c) {
        grid->head[c] = -1;
    }
    for (int i = 0; i < m; ++i) {
        grid_insert(grid, i, X[2 * i], X[2 * i + 1]);
    }
}

static inline void consider_candidate(
    double d2, int idx, int k, int *count_ptr, double *best_d2, int *best_idx
) {
    int count = *count_ptr;
    if (count == k) {
        double worst_d2 = best_d2[k - 1];
        if (d2 > worst_d2 || (d2 == worst_d2 && idx >= best_idx[k - 1])) {
            return;
        }
    } else {
        *count_ptr = count + 1;
    }

    int pos = (count < k) ? count : (k - 1);
    while (pos > 0) {
        double pd2 = best_d2[pos - 1];
        int pidx = best_idx[pos - 1];
        if (d2 < pd2 || (d2 == pd2 && idx < pidx)) {
            best_d2[pos] = pd2;
            best_idx[pos] = pidx;
            pos--;
        } else {
            break;
        }
    }
    best_d2[pos] = d2;
    best_idx[pos] = idx;
}

/* Exact k-nearest-neighbor search in 2-D using expanding Chebyshev shells
 * around (cx, cy)'s cell with provable lower-bound pruning. */
static int find_knn_exact(
    const SpatialGrid *grid,
    const double *X,
    int m,
    double cx,
    double cy,
    int n_neighbors,
    double *best_d2,
    int *best_idx
) {
    if (m <= 0) return 0;
    int k = (n_neighbors < m) ? n_neighbors : m;
    int count = 0;

    if (m <= 96) {
        for (int i = 0; i < m; ++i) {
            double dx = X[2 * i] - cx;
            double dy = X[2 * i + 1] - cy;
            consider_candidate(dx * dx + dy * dy, i, k, &count, best_d2, best_idx);
        }
        return count;
    }

    const int gn = grid->grid_n;
    const double inv_g = grid->inv_g;
    int gx0 = cell_coord(cx, gn);
    int gy0 = cell_coord(cy, gn);

    for (int rad = 0; rad < gn; ++rad) {
        if (rad >= 2 && count == k) {
            double min_shell_d2 = INFINITY;
            if (gx0 - rad >= 0) {
                double d2 = cell_axis_dist2(cx, gx0 - rad, gn, inv_g);
                if (d2 < min_shell_d2) min_shell_d2 = d2;
            }
            if (gx0 + rad < gn) {
                double d2 = cell_axis_dist2(cx, gx0 + rad, gn, inv_g);
                if (d2 < min_shell_d2) min_shell_d2 = d2;
            }
            if (gy0 - rad >= 0) {
                double d2 = cell_axis_dist2(cy, gy0 - rad, gn, inv_g);
                if (d2 < min_shell_d2) min_shell_d2 = d2;
            }
            if (gy0 + rad < gn) {
                double d2 = cell_axis_dist2(cy, gy0 + rad, gn, inv_g);
                if (d2 < min_shell_d2) min_shell_d2 = d2;
            }
            if (min_shell_d2 > best_d2[k - 1]) {
                break;
            }
        }

        int x_lo = gx0 - rad;
        int x_hi = gx0 + rad;
        int y_lo = gy0 - rad;
        int y_hi = gy0 + rad;
        int bx_min = (x_lo < 0) ? 0 : x_lo;
        int bx_max = (x_hi >= gn) ? (gn - 1) : x_hi;
        int by_min = (y_lo < 0) ? 0 : y_lo;
        int by_max = (y_hi >= gn) ? (gn - 1) : y_hi;

        for (int by = by_min; by <= by_max; ++by) {
            double dy_box = cell_axis_dist2(cy, by, gn, inv_g);
            for (int bx = bx_min; bx <= bx_max; ++bx) {
                if (rad > 0 && bx > x_lo && bx < x_hi && by > y_lo && by < y_hi) {
                    bx = x_hi - 1;
                    continue;
                }
                int head = grid->head[by * gn + bx];
                if (head == -1) continue;
                if (count == k) {
                    double box_d2 = dy_box + cell_axis_dist2(cx, bx, gn, inv_g);
                    if (box_d2 > best_d2[k - 1]) continue;
                }
                for (int i = head; i != -1; i = grid->next[i]) {
                    double dx = X[2 * i] - cx;
                    double dy = X[2 * i + 1] - cy;
                    consider_candidate(dx * dx + dy * dy, i, k, &count, best_d2, best_idx);
                }
            }
        }
    }
    return count;
}

typedef struct {
    double usage;
    int idx;
} UsagePair;

static int cmp_usage_desc(const void *a, const void *b) {
    const UsagePair *pa = (const UsagePair *)a;
    const UsagePair *pb = (const UsagePair *)b;
    if (pa->usage > pb->usage) return -1;
    if (pa->usage < pb->usage) return 1;
    /* Match np.argsort(u)[::-1]: reverse index order on exact usage ties */
    return (pb->idx - pa->idx);
}

static int cmp_int_asc(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static int prune_prototypes(
    double *X, double *N, double *usage, int m, int max_prototypes, int a
) {
    if (m <= max_prototypes) return m;
    if (max_prototypes <= 0) return 0;
    UsagePair *pairs = (UsagePair *)malloc((size_t)m * sizeof(UsagePair));
    int *keep = (int *)malloc((size_t)max_prototypes * sizeof(int));
    if (!pairs || !keep) {
        free(pairs);
        free(keep);
        return m;
    }
    for (int i = 0; i < m; ++i) {
        pairs[i].usage = usage[i];
        pairs[i].idx = i;
    }
    qsort(pairs, (size_t)m, sizeof(UsagePair), cmp_usage_desc);
    for (int i = 0; i < max_prototypes; ++i) {
        keep[i] = pairs[i].idx;
    }
    qsort(keep, (size_t)max_prototypes, sizeof(int), cmp_int_asc);

    for (int dst = 0; dst < max_prototypes; ++dst) {
        int src = keep[dst];
        if (dst != src) {
            X[2 * dst]     = X[2 * src];
            X[2 * dst + 1] = X[2 * src + 1];
            memcpy(N + (size_t)dst * (size_t)a,
                   N + (size_t)src * (size_t)a,
                   (size_t)a * sizeof(double));
            usage[dst] = usage[src];
        }
    }
    free(pairs);
    free(keep);
    return max_prototypes;
}

/* -------------------------------------------------------------------------
 * 1. cgr_embed(ids, all_vertices, centre_x, centre_y, r, out_P)
 * ------------------------------------------------------------------------- */
static PyObject *py_cgr_embed(PyObject *self, PyObject *args) {
    Py_buffer ids_buf, verts_buf, out_buf;
    double centre_x, centre_y, r;

    if (!PyArg_ParseTuple(
            args, "y*y*dddw*",
            &ids_buf, &verts_buf, &centre_x, &centre_y, &r, &out_buf)) {
        return NULL;
    }

    Py_ssize_t T = ids_buf.len / (Py_ssize_t)sizeof(int64_t);
    Py_ssize_t n_verts = verts_buf.len / (Py_ssize_t)(2 * sizeof(double));
    if (T < 0 || out_buf.len < T * (Py_ssize_t)(2 * sizeof(double)) || n_verts < 2 ||
        !isfinite(centre_x) || !isfinite(centre_y) || !isfinite(r) || r <= 0.0 || r >= 1.0) {
        PyBuffer_Release(&ids_buf);
        PyBuffer_Release(&verts_buf);
        PyBuffer_Release(&out_buf);
        PyErr_SetString(PyExc_ValueError, "Invalid buffer dimensions or contraction in cgr_embed");
        return NULL;
    }

    const int64_t *ids = (const int64_t *)ids_buf.buf;
    const double *verts = (const double *)verts_buf.buf;
    double *out_P = (double *)out_buf.buf;

    Py_BEGIN_ALLOW_THREADS
    double one_minus = 1.0 - r;
    double acc_x = centre_x;
    double acc_y = centre_y;
    for (Py_ssize_t t = 0; t < T; ++t) {
        int64_t s = ids[t];
        if (s < 0 || s >= n_verts) {
            s = n_verts - 1;
        }
        double rv_x = r * verts[2 * s];
        double rv_y = r * verts[2 * s + 1];
        acc_x = one_minus * acc_x + rv_x;
        acc_y = one_minus * acc_y + rv_y;
        out_P[2 * t]     = acc_x;
        out_P[2 * t + 1] = acc_y;
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&ids_buf);
    PyBuffer_Release(&verts_buf);
    PyBuffer_Release(&out_buf);
    Py_RETURN_NONE;
}

/* -------------------------------------------------------------------------
 * 2. cortex_stream(...)
 * ------------------------------------------------------------------------- */
static PyObject *py_cortex_stream(PyObject *self, PyObject *args) {
    Py_buffer ids_buf, C_buf, X_buf, N_buf, usage_buf, global_buf, bits_buf;
    int a, m, cap, n_neighbors, surprise_growth, max_prototypes, prune_interval, update;
    double dist_threshold, sigma, alpha, background_weight, learn_rate, usage_decay;
    long long steps, surp_n;
    double surp_mean, surp_m2;

    if (!PyArg_ParseTuple(
            args,
            "y*y*w*w*w*w*w*iiiidddddpiidLLddi",
            &ids_buf, &C_buf, &X_buf, &N_buf, &usage_buf, &global_buf, &bits_buf,
            &a, &m, &cap, &n_neighbors,
            &dist_threshold, &sigma, &alpha, &background_weight, &learn_rate,
            &surprise_growth, &max_prototypes, &prune_interval, &usage_decay,
            &steps, &surp_n, &surp_mean, &surp_m2, &update)) {
        return NULL;
    }

    Py_ssize_t T = ids_buf.len / (Py_ssize_t)sizeof(int64_t);
    if (a <= 0 || m < 0 || cap < 1 || cap < m || n_neighbors <= 0 || max_prototypes < 1 ||
        !isfinite(dist_threshold) || dist_threshold < 0.0 ||
        !isfinite(sigma) || sigma <= 0.0 ||
        !isfinite(alpha) || alpha <= 0.0 ||
        !isfinite(background_weight) || background_weight < 0.0 || background_weight > 1.0 ||
        !isfinite(learn_rate) || learn_rate < 0.0 || learn_rate > 1.0 ||
        !isfinite(usage_decay) || usage_decay <= 0.0 || usage_decay > 1.0 ||
        !isfinite(surp_mean) || !isfinite(surp_m2) || surp_m2 < 0.0 ||
        X_buf.len < (Py_ssize_t)cap * (Py_ssize_t)(2 * sizeof(double)) ||
        N_buf.len < (Py_ssize_t)cap * (Py_ssize_t)a * (Py_ssize_t)sizeof(double) ||
        usage_buf.len < (Py_ssize_t)cap * (Py_ssize_t)sizeof(double) ||
        C_buf.len < T * (Py_ssize_t)(2 * sizeof(double)) ||
        bits_buf.len < T * (Py_ssize_t)sizeof(double) ||
        global_buf.len < (Py_ssize_t)a * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&ids_buf);
        PyBuffer_Release(&C_buf);
        PyBuffer_Release(&X_buf);
        PyBuffer_Release(&N_buf);
        PyBuffer_Release(&usage_buf);
        PyBuffer_Release(&global_buf);
        PyBuffer_Release(&bits_buf);
        PyErr_SetString(PyExc_ValueError, "Invalid buffer dimensions or hyperparameters in cortex_stream");
        return NULL;
    }

    const int64_t *ids = (const int64_t *)ids_buf.buf;
    const double *C = (const double *)C_buf.buf;
    double *X = (double *)X_buf.buf;
    double *N = (double *)N_buf.buf;
    double *usage = (double *)usage_buf.buf;
    double *global_counts = (double *)global_buf.buf;
    double *bits = (double *)bits_buf.buf;

    double stack_k_d2[64];
    int stack_k_idx[64];
    double stack_w[64];
    double stack_bg[32];
    double stack_weighted[32];
    double stack_p[32];

    double *best_d2 = (n_neighbors <= 64) ? stack_k_d2 : (double *)malloc((size_t)n_neighbors * sizeof(double));
    int *best_idx = (n_neighbors <= 64) ? stack_k_idx : (int *)malloc((size_t)n_neighbors * sizeof(int));
    double *weights = (n_neighbors <= 64) ? stack_w : (double *)malloc((size_t)n_neighbors * sizeof(double));
    double *bg = (a <= 32) ? stack_bg : (double *)malloc((size_t)a * sizeof(double));
    double *weighted = (a <= 32) ? stack_weighted : (double *)malloc((size_t)a * sizeof(double));
    double *p = (a <= 32) ? stack_p : (double *)malloc((size_t)a * sizeof(double));

    SpatialGrid grid;
    int grid_rc = grid_init(&grid, cap, X, m);
    if (grid_rc != 0 || !best_d2 || !best_idx || !weights || !bg || !weighted || !p) {
        if (n_neighbors > 64) { free(best_d2); free(best_idx); free(weights); }
        if (a > 32) { free(bg); free(weighted); free(p); }
        grid_free(&grid);
        PyBuffer_Release(&ids_buf);
        PyBuffer_Release(&C_buf);
        PyBuffer_Release(&X_buf);
        PyBuffer_Release(&N_buf);
        PyBuffer_Release(&usage_buf);
        PyBuffer_Release(&global_buf);
        PyBuffer_Release(&bits_buf);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    const double decay_factor = (prune_interval > 0) ? pow(usage_decay, (double)prune_interval) : 1.0;
    const int headroom = (prune_interval > 0) ? ((max_prototypes / 4 > 16) ? (max_prototypes / 4) : 16) : 0;
    const int growth_cap = max_prototypes + headroom;

    for (Py_ssize_t t = 0; t < T; ++t) {
        int64_t s = ids[t];
        double cx = C[2 * t];
        double cy = C[2 * t + 1];

        double bg_sum = 0.0;
        for (int c_i = 0; c_i < a; ++c_i) {
            bg_sum += global_counts[c_i];
        }
        for (int c_i = 0; c_i < a; ++c_i) {
            bg[c_i] = global_counts[c_i] / bg_sum;
        }

        int nearest_idx = -1;
        double nearest_dist = INFINITY;

        if (m == 0) {
            for (int c_i = 0; c_i < a; ++c_i) {
                p[c_i] = bg[c_i];
            }
        } else {
            int count = find_knn_exact(&grid, X, m, cx, cy, n_neighbors, best_d2, best_idx);
            nearest_idx = best_idx[0];
            nearest_dist = sqrt(best_d2[0]);

            double wsum = 0.0;
            for (int r_i = 0; r_i < count; ++r_i) {
                double w = exp(-best_d2[r_i] * inv_2sigma2);
                weights[r_i] = w;
                wsum += w;
            }
            if (wsum <= 0.0) {
                for (int c_i = 0; c_i < a; ++c_i) {
                    p[c_i] = bg[c_i];
                }
            } else {
                for (int c_i = 0; c_i < a; ++c_i) {
                    weighted[c_i] = 0.0;
                }
                for (int r_i = 0; r_i < count; ++r_i) {
                    double w = weights[r_i];
                    const double *N_row = N + (size_t)best_idx[r_i] * (size_t)a;
                    for (int c_i = 0; c_i < a; ++c_i) {
                        weighted[c_i] += w * N_row[c_i];
                    }
                }
                double denom = 0.0;
                for (int c_i = 0; c_i < a; ++c_i) {
                    denom += weighted[c_i];
                }
                double p_denom = denom + (double)a * alpha;
                double p_sum = 0.0;
                for (int c_i = 0; c_i < a; ++c_i) {
                    double p_proto = (weighted[c_i] + alpha) / p_denom;
                    double pj = (1.0 - background_weight) * p_proto + background_weight * bg[c_i];
                    p[c_i] = pj;
                    p_sum += pj;
                }
                for (int c_i = 0; c_i < a; ++c_i) {
                    p[c_i] /= p_sum;
                }
            }
        }

        if (s >= 0 && s < a) {
            double ps = p[s];
            double b = (ps > 0.0) ? (-log2(ps)) : 60.0;
            bits[t] = b;

            if (update) {
                surp_n += 1;
                double delta = b - surp_mean;
                surp_mean += delta / (double)surp_n;
                surp_m2 += delta * (b - surp_mean);

                global_counts[s] += 1.0;

                int empty = (m == 0);
                int far = (nearest_dist > dist_threshold);
                int high_surp = 0;
                if (surprise_growth && surp_n >= 50) {
                    double v_denom = (surp_n - 1 > 1) ? (double)(surp_n - 1) : 1.0;
                    double std_v = sqrt(surp_m2 / v_denom);
                    high_surp = (b > surp_mean + 2.0 * std_v);
                }
                int shocked = surprise_growth && high_surp && (nearest_dist > 0.5 * dist_threshold);
                int can_grow = (m < growth_cap);

                if (empty || ((far || shocked) && can_grow)) {
                    if (m < cap) {
                        X[2 * m]     = cx;
                        X[2 * m + 1] = cy;
                        double *N_new = N + (size_t)m * (size_t)a;
                        for (int c_i = 0; c_i < a; ++c_i) {
                            N_new[c_i] = 0.0;
                        }
                        N_new[s] += 1.0;
                        usage[m] = 1.0;
                        grid_insert(&grid, m, cx, cy);
                        m += 1;
                    }
                } else {
                    N[(size_t)nearest_idx * (size_t)a + (size_t)s] += 1.0;
                    usage[nearest_idx] += 1.0;
                    if (learn_rate > 0.0) {
                        double nx = X[2 * nearest_idx]     + learn_rate * (cx - X[2 * nearest_idx]);
                        double ny = X[2 * nearest_idx + 1] + learn_rate * (cy - X[2 * nearest_idx + 1]);
                        X[2 * nearest_idx]     = nx;
                        X[2 * nearest_idx + 1] = ny;
                        grid_move(&grid, nearest_idx, nx, ny);
                    }
                }
            }
        } else {
            bits[t] = NAN;
        }

        if (update) {
            steps += 1;
            if (prune_interval > 0 && (steps % (long long)prune_interval) == 0) {
                for (int i = 0; i < m; ++i) {
                    usage[i] *= decay_factor;
                }
                if (m > max_prototypes) {
                    m = prune_prototypes(X, N, usage, m, max_prototypes, a);
                    grid_rebuild(&grid, X, m);
                }
            }
        }
    }
    if (update && m > max_prototypes) {
        m = prune_prototypes(X, N, usage, m, max_prototypes, a);
    }
    Py_END_ALLOW_THREADS

    if (n_neighbors > 64) { free(best_d2); free(best_idx); free(weights); }
    if (a > 32) { free(bg); free(weighted); free(p); }
    grid_free(&grid);

    PyBuffer_Release(&ids_buf);
    PyBuffer_Release(&C_buf);
    PyBuffer_Release(&X_buf);
    PyBuffer_Release(&N_buf);
    PyBuffer_Release(&usage_buf);
    PyBuffer_Release(&global_buf);
    PyBuffer_Release(&bits_buf);

    return Py_BuildValue("(iLLdd)", m, steps, surp_n, surp_mean, surp_m2);
}

/* -------------------------------------------------------------------------
 * 3. cortex_score_batch(ids, C, X, N, global_counts, bits, a, m, n_neighbors,
 *                       sigma, alpha, background_weight)
 * ------------------------------------------------------------------------- */
static PyObject *py_cortex_score_batch(PyObject *self, PyObject *args) {
    Py_buffer ids_buf, C_buf, X_buf, N_buf, global_buf, bits_buf;
    int a, m, n_neighbors;
    double sigma, alpha, background_weight;

    if (!PyArg_ParseTuple(
            args,
            "y*y*y*y*y*w*iiiddd",
            &ids_buf, &C_buf, &X_buf, &N_buf, &global_buf, &bits_buf,
            &a, &m, &n_neighbors, &sigma, &alpha, &background_weight)) {
        return NULL;
    }

    Py_ssize_t T = ids_buf.len / (Py_ssize_t)sizeof(int64_t);
    if (a <= 0 || n_neighbors <= 0 || m < 0 ||
        !isfinite(sigma) || sigma <= 0.0 ||
        !isfinite(alpha) || alpha <= 0.0 ||
        !isfinite(background_weight) || background_weight < 0.0 || background_weight > 1.0 ||
        X_buf.len < (Py_ssize_t)m * (Py_ssize_t)(2 * sizeof(double)) ||
        N_buf.len < (Py_ssize_t)m * (Py_ssize_t)a * (Py_ssize_t)sizeof(double) ||
        C_buf.len < T * (Py_ssize_t)(2 * sizeof(double)) ||
        bits_buf.len < T * (Py_ssize_t)sizeof(double) ||
        global_buf.len < (Py_ssize_t)a * (Py_ssize_t)sizeof(double)) {
        PyBuffer_Release(&ids_buf);
        PyBuffer_Release(&C_buf);
        PyBuffer_Release(&X_buf);
        PyBuffer_Release(&N_buf);
        PyBuffer_Release(&global_buf);
        PyBuffer_Release(&bits_buf);
        PyErr_SetString(PyExc_ValueError, "Invalid buffer dimensions or hyperparameters in cortex_score_batch");
        return NULL;
    }

    const int64_t *ids = (const int64_t *)ids_buf.buf;
    const double *C = (const double *)C_buf.buf;
    const double *X = (const double *)X_buf.buf;
    const double *N = (const double *)N_buf.buf;
    const double *global_counts = (const double *)global_buf.buf;
    double *bits = (double *)bits_buf.buf;

    double stack_k_d2[64];
    int stack_k_idx[64];
    double stack_w[64];
    double stack_bg[32];
    double stack_weighted[32];
    double stack_p[32];

    double *best_d2 = (n_neighbors <= 64) ? stack_k_d2 : (double *)malloc((size_t)n_neighbors * sizeof(double));
    int *best_idx = (n_neighbors <= 64) ? stack_k_idx : (int *)malloc((size_t)n_neighbors * sizeof(int));
    double *weights = (n_neighbors <= 64) ? stack_w : (double *)malloc((size_t)n_neighbors * sizeof(double));
    double *bg = (a <= 32) ? stack_bg : (double *)malloc((size_t)a * sizeof(double));
    double *weighted = (a <= 32) ? stack_weighted : (double *)malloc((size_t)a * sizeof(double));
    double *p = (a <= 32) ? stack_p : (double *)malloc((size_t)a * sizeof(double));

    SpatialGrid grid;
    int grid_rc = grid_init(&grid, m > 0 ? m : 1, X, m);
    if (grid_rc != 0 || !best_d2 || !best_idx || !weights || !bg || !weighted || !p) {
        if (n_neighbors > 64) { free(best_d2); free(best_idx); free(weights); }
        if (a > 32) { free(bg); free(weighted); free(p); }
        grid_free(&grid);
        PyBuffer_Release(&ids_buf);
        PyBuffer_Release(&C_buf);
        PyBuffer_Release(&X_buf);
        PyBuffer_Release(&N_buf);
        PyBuffer_Release(&global_buf);
        PyBuffer_Release(&bits_buf);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    double bg_sum = 0.0;
    for (int c_i = 0; c_i < a; ++c_i) bg_sum += global_counts[c_i];
    for (int c_i = 0; c_i < a; ++c_i) bg[c_i] = global_counts[c_i] / bg_sum;
    const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    for (Py_ssize_t t = 0; t < T; ++t) {
        int64_t s = ids[t];
        if (s < 0 || s >= a) {
            bits[t] = NAN;
            continue;
        }
        if (m == 0) {
            double ps = bg[s];
            if (ps < 1e-18) ps = 1e-18;
            else if (ps > 1.0) ps = 1.0;
            bits[t] = -log2(ps);
            continue;
        }

        double cx = C[2 * t];
        double cy = C[2 * t + 1];
        int count = find_knn_exact(&grid, X, m, cx, cy, n_neighbors, best_d2, best_idx);

        double wsum = 0.0;
        for (int r_i = 0; r_i < count; ++r_i) {
            double w = exp(-best_d2[r_i] * inv_2sigma2);
            weights[r_i] = w;
            wsum += w;
        }
        if (wsum <= 0.0) {
            double ps = bg[s];
            if (ps < 1e-18) ps = 1e-18;
            else if (ps > 1.0) ps = 1.0;
            bits[t] = -log2(ps);
            continue;
        }

        for (int c_i = 0; c_i < a; ++c_i) weighted[c_i] = 0.0;
        for (int r_i = 0; r_i < count; ++r_i) {
            double w = weights[r_i];
            const double *N_row = N + (size_t)best_idx[r_i] * (size_t)a;
            for (int c_i = 0; c_i < a; ++c_i) {
                weighted[c_i] += w * N_row[c_i];
            }
        }
        double denom = 0.0;
        for (int c_i = 0; c_i < a; ++c_i) denom += weighted[c_i];
        double p_denom = denom + (double)a * alpha;
        double p_sum = 0.0;
        for (int c_i = 0; c_i < a; ++c_i) {
            double p_proto = (weighted[c_i] + alpha) / p_denom;
            double pj = (1.0 - background_weight) * p_proto + background_weight * bg[c_i];
            p[c_i] = pj;
            p_sum += pj;
        }
        double ps = p[s] / p_sum;
        if (ps < 1e-18) ps = 1e-18;
        else if (ps > 1.0) ps = 1.0;
        bits[t] = -log2(ps);
    }
    Py_END_ALLOW_THREADS

    if (n_neighbors > 64) { free(best_d2); free(best_idx); free(weights); }
    if (a > 32) { free(bg); free(weighted); free(p); }
    grid_free(&grid);

    PyBuffer_Release(&ids_buf);
    PyBuffer_Release(&C_buf);
    PyBuffer_Release(&X_buf);
    PyBuffer_Release(&N_buf);
    PyBuffer_Release(&global_buf);
    PyBuffer_Release(&bits_buf);
    Py_RETURN_NONE;
}

static PyMethodDef FastMethods[] = {
    {"cgr_embed", py_cgr_embed, METH_VARARGS,
     "Compute CGR trajectory points in-place over contiguous buffers."},
    {"cortex_stream", py_cortex_stream, METH_VARARGS,
     "Stream online learning or scoring over a CGR context trajectory."},
    {"cortex_score_batch", py_cortex_score_batch, METH_VARARGS,
     "Score a sequence in batch mode without modifying prototypes."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastmodule = {
    PyModuleDef_HEAD_INIT,
    "_fast",
    "Native C spatial-grid accelerator for GenoCortex.",
    -1,
    FastMethods
};

PyMODINIT_FUNC PyInit__fast(void) {
    return PyModule_Create(&fastmodule);
}
