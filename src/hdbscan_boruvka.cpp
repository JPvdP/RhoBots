// hdbscan_boruvka.cpp
//
// Scalable HDBSCAN via Borůvka minimum spanning tree on the mutual-reachability
// kNN graph.  Memory cost is O(n × k) — avoids the O(n²) distance matrix that
// dbscan::hdbscan() builds internally and that causes OOM at ~30K+ documents.
//
// ALGORITHM OUTLINE
// -----------------
// 1. Core distances  — knn_dist[i, min_pts-1] = distance to the min_pts-th NN.
// 2. Borůvka MST     — iterative "cheapest edge per component" using the mutual
//                      reachability metric d_mr(i,j) = max(core[i], core[j], d(i,j)).
//                      O(n × k × log n) time, O(n × k) memory.
// 3. SL tree         — build single-linkage hierarchy from sorted MST edges.
//                      Internal node IDs: n..n+n_merges-1.
// 4. Condense tree   — top-down pass: mark each node with a condensed cluster ID.
//                      Merges where one side < min_pts: small side falls out as noise;
//                      cluster continues through the large side.
//                      Merges where both sides >= min_pts: true split; two new clusters.
//                      Stability is accumulated as lambda × n_exiting_points at each exit.
// 5. EOM extraction  — bottom-up stability propagation; select clusters that maximise
//                      total stability.
// 6. Label points    — walk node_cluster[] to find each point's selected cluster.
//
// REFERENCE
// ---------
// Campello, Moulavi, Sander (2013). Density-Based Clustering Based on Hierarchical
// Density Estimates. PAKDD. doi:10.1007/978-3-642-37456-2_14.

#include <Rcpp.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>

using namespace Rcpp;
using std::vector;

// =============================================================================
// Union-Find (path compression + union by size)
// =============================================================================

struct UF {
  vector<int> par, sz;
  int nc;

  UF(int n) : par(n), sz(n, 1), nc(n) {
    std::iota(par.begin(), par.end(), 0);
  }

  int find(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
  }

  // Returns new root after merge, or -1 if already in same component.
  int unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return -1;
    if (sz[a] < sz[b]) std::swap(a, b);
    par[b] = a; sz[a] += sz[b]; --nc;
    return a;
  }

  int size(int x) { return sz[find(x)]; }
};

// =============================================================================
// Borůvka MST on the mutual-reachability kNN graph
// =============================================================================

struct Edge { double w; int u, v; };

static vector<Edge> boruvka_mst(
    const IntegerMatrix& idx,   // n × k, 1-indexed
    const NumericMatrix& dist,  // n × k
    const vector<double>& core,
    int n, int k)
{
  vector<Edge> mst; mst.reserve(n - 1);
  UF uf(n);
  vector<double> bw(n); vector<int> bu(n), bv(n);

  while ((int)mst.size() < n - 1 && uf.nc > 1) {
    std::fill(bw.begin(), bw.end(), std::numeric_limits<double>::infinity());
    std::fill(bu.begin(), bu.end(), -1);

    for (int i = 0; i < n; i++) {
      int ci = uf.find(i);
      for (int ki = 0; ki < k; ki++) {
        int j = idx(i, ki) - 1;                      // convert to 0-indexed
        if (j < 0 || j >= n) continue;
        int cj = uf.find(j);
        if (ci == cj) continue;
        double d   = dist(i, ki);
        double mrd = std::max(std::max(core[i], core[j]), d);
        if (mrd < bw[ci]) { bw[ci] = mrd; bu[ci] = i; bv[ci] = j; }
      }
    }

    bool any = false;
    for (int r = 0; r < n; r++) {
      if (bu[r] < 0) continue;
      if (uf.unite(bu[r], bv[r]) >= 0) {
        mst.push_back({bw[r], bu[r], bv[r]});
        any = true;
        if ((int)mst.size() == n - 1) break;
      }
    }
    if (!any) break;   // kNN graph is disconnected; remaining points stay as noise
  }

  std::sort(mst.begin(), mst.end(),
            [](const Edge& a, const Edge& b){ return a.w < b.w; });
  return mst;
}

// =============================================================================
// Single-linkage tree
// Node IDs: 0..n-1 = leaves (points), n..n+nm-1 = internal merge nodes.
// =============================================================================

struct SLNode { int left, right, size; double lambda; };

static vector<SLNode> build_sl(const vector<Edge>& mst, int n) {
  int nm = (int)mst.size();
  vector<SLNode> sl(nm);
  vector<int> cnode(n); std::iota(cnode.begin(), cnode.end(), 0);
  vector<int> csz(n, 1);
  UF uf(n);

  for (int i = 0; i < nm; i++) {
    int u = mst[i].u, v = mst[i].v;
    double lam = mst[i].w > 0.0 ? 1.0 / mst[i].w : 1e15;
    int ru = uf.find(u), rv = uf.find(v);
    int nl = cnode[ru], nr = cnode[rv], sz2 = csz[ru] + csz[rv];
    uf.unite(u, v);
    int nr2 = uf.find(u);
    cnode[nr2] = n + i;
    csz[nr2]   = sz2;
    sl[i] = {nl, nr, sz2, lam};
  }
  return sl;
}

// Helper: size of a node (leaf = 1, internal = sl[node-n].size)
static inline int nd_sz(int node, const vector<SLNode>& sl, int n) {
  return node < n ? 1 : sl[node - n].size;
}

// =============================================================================
// Main exported function
// =============================================================================

//' HDBSCAN via Boruvka MST on mutual-reachability kNN graph
//'
//' Internal function called by \code{cluster_docs.hdbscan_clustering()}.
//' Returns a list with \code{labels} (integer vector, 0 = noise) and
//' \code{n_mst_edges} (number of MST edges built).  When \code{n_mst_edges < n-1}
//' the kNN graph was disconnected; the R wrapper retries with larger k.
//'
//' @param knn_idx Integer matrix (n × k), 1-indexed nearest-neighbour indices.
//' @param knn_dist Numeric matrix (n × k), corresponding distances.
//' @param min_pts Integer minimum cluster size / core-distance order.
//' @return Named list: \code{labels} (IntegerVector), \code{n_mst_edges} (int).
// [[Rcpp::export]]
List hdbscan_boruvka_cpp(IntegerMatrix knn_idx,
                          NumericMatrix knn_dist,
                          int           min_pts)
{
  const int n = knn_idx.nrow();
  const int k = knn_idx.ncol();

  // --- 1. Core distances ---------------------------------------------------
  const int cc = std::min(min_pts - 1, k - 1);
  vector<double> core(n);
  for (int i = 0; i < n; i++) core[i] = knn_dist(i, cc);

  // --- 2. Borůvka MST ------------------------------------------------------
  vector<Edge> mst = boruvka_mst(knn_idx, knn_dist, core, n, k);
  const int nm = (int)mst.size();

  if (nm == 0)
    return List::create(Named("labels") = IntegerVector(n, 0),
                        Named("n_mst_edges") = 0);

  // --- 3. Single-linkage tree ----------------------------------------------
  vector<SLNode> sl = build_sl(mst, n);

  // node_cluster[i] = condensed cluster ID for node i (-1 = noise/dissolved)
  // node_lb[i]      = lambda_birth of that cluster
  const int ntot = n + nm;
  vector<int>    nc(ntot, -1);
  vector<double> nb(ntot, 0.0);

  // Cluster registry (grown dynamically during top-down pass)
  vector<double>      cl_stab;      // accumulated stability per cluster
  vector<int>         cl_par;       // parent cluster ID (-1 for root)
  vector<vector<int>> cl_ch;        // child cluster IDs (only at true splits)

  // Parents always have lower IDs than children (created in top-down order).
  int next_cl = 0;

  auto new_cl = [&](int par, double lb) -> int {
    int id = next_cl++;
    cl_stab.push_back(0.0);
    cl_par.push_back(par);
    cl_ch.push_back(vector<int>());
    if (par >= 0) cl_ch[par].push_back(id);
    (void)lb;   // lb stored per-node in nb[], not per-cluster
    return id;
  };

  // Assign root cluster to the top-level merge node
  nc[n + nm - 1] = new_cl(-1, 0.0);
  nb[n + nm - 1] = 0.0;

  // --- 4. Condense tree (top-down: m = nm-1 … 0) --------------------------
  for (int m = nm - 1; m >= 0; m--) {
    const int    nid = n + m;
    const int    cl  = nc[nid];
    if (cl < 0) continue;

    const double lam = sl[m].lambda;
    const double lb  = nb[nid];
    const int    lft = sl[m].left,  ls = nd_sz(lft, sl, n);
    const int    rgt = sl[m].right, rs = nd_sz(rgt, sl, n);
    const bool   lb_ = ls >= min_pts;
    const bool   rb_ = rs >= min_pts;

    if (lb_ && rb_) {
      // True split: all (ls + rs) points exit cl at lam
      cl_stab[cl] += (double)(ls + rs) * (lam - lb);
      int cl_l = new_cl(cl, lam), cl_r = new_cl(cl, lam);
      nc[lft] = cl_l; nb[lft] = lam;
      nc[rgt] = cl_r; nb[rgt] = lam;

    } else if (lb_) {
      // Right falls out as noise
      cl_stab[cl] += (double)rs * (lam - lb);
      nc[lft] = cl;   nb[lft] = lb;
      nc[rgt] = -1;

    } else if (rb_) {
      // Left falls out as noise
      cl_stab[cl] += (double)ls * (lam - lb);
      nc[lft] = -1;
      nc[rgt] = cl;   nb[rgt] = lb;

    } else {
      // Both sides too small: cluster dissolves, all points become noise
      cl_stab[cl] += (double)(ls + rs) * (lam - lb);
      nc[lft] = -1;
      nc[rgt] = -1;
    }
  }

  const int ncl = next_cl;
  if (ncl == 0)
    return List::create(Named("labels") = IntegerVector(n, 0),
                        Named("n_mst_edges") = nm);

  // --- 5. EOM cluster extraction -------------------------------------------
  // Bottom-up pass (descending ID = leaves before parents):
  // sel_stab[cl] = max(cl's own stability, sum of children's sel_stab)
  vector<double> sel_stab(ncl);
  vector<bool>   sel_self(ncl, false);
  for (int i = 0; i < ncl; i++) sel_stab[i] = cl_stab[i];

  for (int cl = ncl - 1; cl >= 0; cl--) {
    if (cl_ch[cl].empty()) {
      sel_self[cl] = true;          // leaf cluster: always select itself
    } else {
      double csum = 0.0;
      for (int ch : cl_ch[cl]) csum += sel_stab[ch];
      if (cl_stab[cl] >= csum) {
        sel_self[cl]  = true;
        sel_stab[cl]  = cl_stab[cl];
      } else {
        sel_self[cl]  = false;
        sel_stab[cl]  = csum;
      }
    }
  }

  // Top-down pass (ascending ID = root before children):
  // propagate "active" flag; a cluster is selected iff it is active AND sel_self.
  vector<bool> active(ncl, false), selected(ncl, false);
  active[0] = true;

  for (int cl = 0; cl < ncl; cl++) {
    if (!active[cl]) continue;
    if (sel_self[cl]) {
      selected[cl] = true;
      // children not activated → they are subsumed
    } else {
      for (int ch : cl_ch[cl]) active[ch] = true;
    }
  }

  // --- 6. Representative: lowest selected ancestor for each cluster ---------
  // Processed ascending (parents before children → par < cl guaranteed).
  vector<int> rep(ncl, -1);
  for (int cl = 0; cl < ncl; cl++) {
    if (selected[cl]) {
      rep[cl] = cl;
    } else {
      int par = cl_par[cl];
      rep[cl] = (par >= 0) ? rep[par] : -1;
    }
  }

  // --- 7. Map selected clusters to 1-indexed labels (0 = noise) ------------
  vector<int> lmap(ncl, 0);
  int next_lbl = 1;
  for (int cl = 0; cl < ncl; cl++) if (selected[cl]) lmap[cl] = next_lbl++;

  // --- 8. SL parent pointers (leaves point to their merge node) ------------
  // nc[internal_node] is always valid (set during top-down pass).
  // nc[leaf_point] may be -1 when the point was on the small side of a split.
  // Walking up via sl_par handles both cases: we find the first ancestor whose
  // cluster is valid, then resolve it through rep[].
  vector<int> sl_par(n + nm, -1);
  for (int m = 0; m < nm; m++) {
    int nid = n + m;
    sl_par[sl[m].left]  = nid;
    sl_par[sl[m].right] = nid;
  }

  // --- 9. Assign point labels via parent-walk ------------------------------
  IntegerVector labels(n, 0);
  for (int i = 0; i < n; i++) {
    int node = i;
    while (node != -1) {
      int cl = nc[node];
      if (cl >= 0) {
        int r = rep[cl];
        if (r >= 0) labels[i] = lmap[r];
        break;
      }
      node = sl_par[node];
    }
  }

  return List::create(Named("labels") = labels, Named("n_mst_edges") = nm);
}
