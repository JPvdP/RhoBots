// hdbscan_boruvka.cpp
//
// Scalable HDBSCAN via Borůvka minimum spanning tree on the mutual-reachability
// kNN graph.  Memory cost is O(n × k) — avoids the O(n²) distance matrix that
// dbscan::hdbscan() builds internally and that causes OOM at ~30K+ documents.
//
// ─────────────────────────────────────────────────────────────────────────────
// BACKGROUND: WHY A CUSTOM IMPLEMENTATION?
// ─────────────────────────────────────────────────────────────────────────────
// The standard HDBSCAN implementation in R's dbscan package uses Prim's
// algorithm to build the minimum spanning tree.  Prim's is fine algorithmically
// but requires the FULL n×n mutual-reachability distance matrix to be held in
// memory so it can find the globally cheapest edge at every step.  For n=30,000
// documents that is 30,000² × 8 bytes ≈ 7 GB — too large for most laptops.
//
// Python's hdbscan package avoids this with the Borůvka algorithm, which only
// ever looks at each point's k nearest neighbours, keeping memory at O(n × k).
// We implement the same idea here in C++ via Rcpp so it can be called from R.
//
// ─────────────────────────────────────────────────────────────────────────────
// ALGORITHM OUTLINE (read this before the code)
// ─────────────────────────────────────────────────────────────────────────────
//
// Step 1 — Core distances
//   The core distance of point i is its distance to its min_pts-th nearest
//   neighbour.  Intuitively: it measures how "lonely" point i is.  A point
//   deep inside a dense cluster has a small core distance; an isolated point
//   far from any neighbours has a large core distance.
//
// Step 2 — Mutual reachability distance
//   HDBSCAN does not cluster in the raw distance space.  Instead it uses the
//   mutual reachability distance:
//
//     d_mr(i, j) = max( core(i), core(j), d(i, j) )
//
//   This transformation "smooths out" density differences: two points that are
//   both inside dense regions retain their actual distance, but a sparse point
//   is pushed farther from its neighbours so that sparse regions are less likely
//   to generate spurious clusters.
//
// Step 3 — Borůvka minimum spanning tree
//   We build the MST of the complete mutual-reachability graph, but we only
//   consider edges that appear in the pre-computed kNN graph (i.e., pairs (i,j)
//   where j is one of i's k nearest neighbours).  This restricts us to O(n×k)
//   candidate edges instead of O(n²).
//
//   Borůvka's algorithm (1926) works in rounds:
//     - Start: every point is its own component (n components total).
//     - Each round: for every component, find the cheapest edge leaving it.
//     - Add those edges, merge the components they connect.
//     - Repeat until one component remains (= a spanning tree with n-1 edges).
//   Each round at least halves the number of components, so at most log₂(n)
//   rounds are needed.
//
//   We track components with a Union-Find data structure (see below).
//
// Step 4 — Single-linkage tree (dendrogram)
//   Sort the n-1 MST edges by weight (ascending = closest merges first).
//   Process them in order, merging components.  Each merge creates an internal
//   node in a binary tree.  Node IDs 0..n-1 are the original data points
//   (leaves); internal nodes get IDs n, n+1, ..., n+nm-1 where nm = n-1.
//   The weight of an internal node is converted to lambda = 1/weight (so that
//   higher lambda = tighter = more "clustery").
//
// Step 5 — Condensed cluster tree
//   The single-linkage tree has n-1 internal nodes.  Most of them represent
//   minor consolidations within a developing cluster rather than true splits
//   into two coherent groups.  The condensed tree simplifies this by tracking
//   only "significant" events:
//
//     - TRUE SPLIT: both children have ≥ min_pts points.
//       → Two new clusters are born.  The parent cluster "ends" at this lambda.
//
//     - ONE SIDE TOO SMALL: one child has < min_pts points.
//       → The small child "falls out" as noise at this lambda.
//         The cluster continues through the large child unchanged.
//
//     - BOTH TOO SMALL: both children have < min_pts points.
//       → The cluster dissolves.  All remaining points become noise.
//
//   Cluster stability is accumulated at every step.  The stability of a cluster
//   C born at lambda_birth is:
//
//     S(C) = Σ_p  (lambda_exit(p) - lambda_birth(C))
//
//   where lambda_exit(p) is the lambda at which point p leaves C (either as
//   noise or because C splits).  Intuitively: stability rewards clusters that
//   persist over a wide range of density levels.
//
// Step 6 — Excess-of-Mass (EOM) cluster extraction
//   After the condensed tree is built, we choose WHICH clusters to report.
//   EOM selects the set of clusters that maximises total stability:
//
//     - Bottom-up pass: for each cluster, compare its own stability to the
//       sum of the stabilities of its children.  sel_stab[C] = max of these.
//     - Top-down pass: starting from the root, propagate an "active" flag.
//       A cluster is selected iff it is active AND it preferred its own
//       stability over its children's.
//
//   The result is a set of non-overlapping clusters that together explain the
//   density structure of the data as well as possible.
//
// Step 7 — Label points
//   Each point walks up the single-linkage tree to find the first ancestor
//   node whose condensed cluster is selected.  If found → assign that cluster
//   label.  If not → the point is noise (label 0).
//
// ─────────────────────────────────────────────────────────────────────────────
// REFERENCE
// ─────────────────────────────────────────────────────────────────────────────
// Campello, Moulavi, Sander (2013). Density-Based Clustering Based on
// Hierarchical Density Estimates. PAKDD.
// doi:10.1007/978-3-642-37456-2_14

#include <Rcpp.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <limits>

using namespace Rcpp;
using std::vector;

// =============================================================================
// Union-Find data structure
// =============================================================================
//
// A Union-Find (also called Disjoint Set Union, DSU) tracks a partition of
// n elements into disjoint sets.  We need it to answer two questions quickly:
//
//   find(x)       → which component does x belong to? (returns a representative)
//   unite(a, b)   → merge the components containing a and b.
//
// Naively, find() could walk a chain of "parent" pointers up to the root.
// Two optimisations make both operations nearly O(1) in practice:
//
//   Path compression (in find):
//     While walking to the root, reattach every node directly to its grandparent
//     (par[x] = par[par[x]]).  This flattens the tree over time.
//
//   Union by size (in unite):
//     Always attach the smaller tree under the larger one.  This keeps trees
//     shallow and prevents the worst-case O(n) chain.
//
// Together these give amortised O(α(n)) per operation, where α is the
// inverse Ackermann function — effectively constant for all practical n.

struct UF {
  vector<int> par;   // par[x] = parent of x (par[root] = root)
  vector<int> sz;    // sz[x]  = size of the subtree rooted at x
  int nc;            // current number of distinct components

  // Constructor: initialise n singletons {0}, {1}, ..., {n-1}.
  UF(int n) : par(n), sz(n, 1), nc(n) {
    std::iota(par.begin(), par.end(), 0);   // par[i] = i (each is its own root)
  }

  // find(x): return the root of x's component, with path-halving compression.
  int find(int x) {
    while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
    return x;
  }

  // unite(a, b): merge the components containing a and b.
  // Returns the new root (the representative of the merged component),
  // or -1 if a and b were already in the same component (nothing to do).
  int unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return -1;            // same component → no merge needed
    if (sz[a] < sz[b]) std::swap(a, b);   // attach smaller (b) under larger (a)
    par[b] = a;
    sz[a] += sz[b];
    --nc;                             // one fewer component
    return a;
  }

  // size(x): how many elements are in x's component?
  int size(int x) { return sz[find(x)]; }
};

// =============================================================================
// Borůvka MST on the mutual-reachability kNN graph
// =============================================================================
//
// Input:
//   idx   — n × k integer matrix of neighbour indices (1-indexed, from dbscan::kNN)
//   dist  — n × k numeric matrix of corresponding Euclidean distances
//   core  — vector of length n; core[i] = dist to min_pts-th neighbour of i
//   n     — number of points
//   k     — number of nearest neighbours
//
// Output:
//   A vector of MST edges sorted by ascending mutual reachability weight.
//   In a fully connected graph this has exactly n-1 edges.  If the kNN graph
//   is disconnected (clusters far apart with small k), fewer edges are returned.

struct Edge { double w; int u, v; };

static vector<Edge> boruvka_mst(
    const IntegerMatrix& idx,
    const NumericMatrix& dist,
    const vector<double>& core,
    int n, int k)
{
  vector<Edge> mst;
  mst.reserve(n - 1);   // at most n-1 edges in a spanning tree

  UF uf(n);             // start: every point is its own component

  // Per-component "best outgoing edge": bw = weight, bu/bv = endpoints.
  // These are indexed by component ROOT (not by arbitrary element).
  vector<double> bw(n);
  vector<int>    bu(n), bv(n);

  // Keep iterating as long as the tree is incomplete and there are ≥ 2 components.
  while ((int)mst.size() < n - 1 && uf.nc > 1) {

    // ── Phase A: find the cheapest outgoing edge per component ────────────────
    // Reset: every component starts with "no candidate edge yet".
    std::fill(bw.begin(), bw.end(), std::numeric_limits<double>::infinity());
    std::fill(bu.begin(), bu.end(), -1);

    for (int i = 0; i < n; i++) {
      int ci = uf.find(i);   // which component does i belong to?

      for (int ki = 0; ki < k; ki++) {
        int j = idx(i, ki) - 1;    // convert R's 1-based index to 0-based
        if (j < 0 || j >= n) continue;

        int cj = uf.find(j);
        if (ci == cj) continue;    // i and j are already in the same component

        // Mutual reachability distance: d_mr(i,j) = max(core[i], core[j], d(i,j))
        // This is the weight HDBSCAN assigns to the edge between i and j.
        double d   = dist(i, ki);
        double mrd = std::max(std::max(core[i], core[j]), d);

        // Update the best outgoing edge for component ci.
        if (mrd < bw[ci]) { bw[ci] = mrd; bu[ci] = i; bv[ci] = j; }
      }
    }

    // ── Phase B: add the best edges to the MST ────────────────────────────────
    // Iterate over all component roots.  If a component found an outgoing edge,
    // try to add it.  We call unite() rather than blindly adding to avoid
    // duplicates: if components A and B both picked the same edge, the first
    // unite() call merges them and the second returns -1 (already same component).
    bool any = false;
    for (int r = 0; r < n; r++) {
      if (bu[r] < 0) continue;                     // no candidate edge for component r
      if (uf.unite(bu[r], bv[r]) >= 0) {           // merge succeeded → new MST edge
        mst.push_back({bw[r], bu[r], bv[r]});
        any = true;
        if ((int)mst.size() == n - 1) break;        // spanning tree complete → done
      }
    }

    // If no edge was added this round, the remaining components are unreachable
    // from each other within the kNN graph (graph is disconnected).
    // The R wrapper detects this via nm < n-1 and retries with a larger k.
    if (!any) break;
  }

  // Sort edges by weight so the single-linkage tree is built in merge order.
  std::sort(mst.begin(), mst.end(),
            [](const Edge& a, const Edge& b){ return a.w < b.w; });
  return mst;
}

// =============================================================================
// Single-linkage tree (dendrogram)
// =============================================================================
//
// Process MST edges in order of ascending weight (= ascending merge distance
// = descending lambda, since lambda = 1/weight).
//
// Node numbering:
//   Leaves (original data points): IDs 0 .. n-1
//   Internal merge nodes:           IDs n .. n+nm-1
//   The root is always n+nm-1 (the last merge).
//
// Each internal node records:
//   left, right — the two subtrees being merged (can be leaves or other nodes)
//   size        — total number of leaves below (= size of the merged component)
//   lambda      — the density level at which this merge occurs (= 1/edge_weight)
//
// We use a second Union-Find to track which internal node currently represents
// each component's "top" (cnode[root] = the highest-level node in that component).

struct SLNode { int left, right, size; double lambda; };

static vector<SLNode> build_sl(const vector<Edge>& mst, int n) {
  int nm = (int)mst.size();
  vector<SLNode> sl(nm);

  // cnode[r] = the SL-tree node ID that currently represents component root r.
  // Initially every point is its own node (leaf IDs 0..n-1).
  vector<int> cnode(n); std::iota(cnode.begin(), cnode.end(), 0);
  vector<int> csz(n, 1);
  UF uf(n);

  for (int i = 0; i < nm; i++) {
    int u = mst[i].u, v = mst[i].v;

    // lambda = 1/distance.  If the distance is exactly 0 (two identical points),
    // use a large finite value instead of infinity to avoid numerical issues.
    double lam = mst[i].w > 0.0 ? 1.0 / mst[i].w : 1e15;

    // Which component does each endpoint currently belong to?
    int ru = uf.find(u), rv = uf.find(v);

    // Retrieve the current "top" SL node for each component.
    int nl = cnode[ru], nr = cnode[rv];
    int sz2 = csz[ru] + csz[rv];

    // Merge the two components in our Union-Find.
    uf.unite(u, v);
    int nr2 = uf.find(u);    // root of the newly merged component

    // The new internal SL node (ID = n+i) becomes the "top" of the merged component.
    cnode[nr2] = n + i;
    csz[nr2]   = sz2;

    sl[i] = {nl, nr, sz2, lam};
  }
  return sl;
}

// Helper: how many leaves does a given SL-tree node contain?
// Leaves have ID < n, so they contribute 1; internal nodes store their size.
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
//' @param knn_idx Integer matrix (n x k), 1-indexed nearest-neighbour indices.
//' @param knn_dist Numeric matrix (n x k), corresponding distances.
//' @param min_pts Integer minimum cluster size / core-distance order.
//' @return Named list: \code{labels} (IntegerVector), \code{n_mst_edges} (int).
//' @keywords internal
// [[Rcpp::export]]
List hdbscan_boruvka_cpp(IntegerMatrix knn_idx,
                          NumericMatrix knn_dist,
                          int           min_pts)
{
  const int n = knn_idx.nrow();   // number of documents
  const int k = knn_idx.ncol();   // number of nearest neighbours per document

  // ── Step 1: Core distances ────────────────────────────────────────────────
  // core[i] = distance from point i to its min_pts-th nearest neighbour.
  // knn_dist is sorted (column 0 = nearest, column k-1 = furthest), so the
  // min_pts-th neighbour is at column index min_pts-1.
  // We clamp to k-1 in case k < min_pts (shouldn't happen in practice, but safe).
  const int cc = std::min(min_pts - 1, k - 1);
  vector<double> core(n);
  for (int i = 0; i < n; i++) core[i] = knn_dist(i, cc);

  // ── Step 2: Borůvka MST on mutual-reachability kNN graph ─────────────────
  vector<Edge> mst = boruvka_mst(knn_idx, knn_dist, core, n, k);
  const int nm = (int)mst.size();

  // If no edges were added (completely isolated points), return all noise.
  if (nm == 0)
    return List::create(Named("labels") = IntegerVector(n, 0),
                        Named("n_mst_edges") = 0);

  // ── Step 3: Single-linkage tree ───────────────────────────────────────────
  vector<SLNode> sl = build_sl(mst, n);

  // ── Step 4: Condensed cluster tree ────────────────────────────────────────
  // nc[node]  = condensed cluster ID assigned to this SL-tree node (-1 = no cluster)
  // nb[node]  = lambda_birth of the cluster that currently owns this node
  const int ntot = n + nm;
  vector<int>    nc(ntot, -1);
  vector<double> nb(ntot, 0.0);

  // The condensed tree uses its own cluster IDs (separate from SL node IDs).
  // We grow three parallel arrays as new clusters are created:
  vector<double>      cl_stab;           // S(C): accumulated stability of each cluster
  vector<int>         cl_par;            // parent cluster ID in the condensed tree (-1 = root)
  vector<vector<int>> cl_ch;             // child cluster IDs (non-empty only at true splits)

  // Cluster IDs are assigned in creation order.  Because we process the SL tree
  // top-down (largest m first), parents always receive lower IDs than children.
  // This is crucial: it lets us process clusters in ascending ID order during EOM
  // and be guaranteed that a parent's rep[] is already computed when we reach a child.
  int next_cl = 0;

  // Lambda that creates a new cluster in the condensed tree and registers it.
  auto new_cl = [&](int par, double lb) -> int {
    int id = next_cl++;
    cl_stab.push_back(0.0);
    cl_par.push_back(par);
    cl_ch.push_back(vector<int>());
    if (par >= 0) cl_ch[par].push_back(id);   // register as child of parent
    (void)lb;   // lambda_birth stored per-node in nb[], not per-cluster
    return id;
  };

  // The root SL node (n+nm-1) represents the entire dataset.
  // Assign it to a new root cluster born at lambda = 0.
  nc[n + nm - 1] = new_cl(-1, 0.0);
  nb[n + nm - 1] = 0.0;

  // Process SL tree nodes from root downwards (high m → low m).
  // At each merge node nid = n+m:
  //   lft, rgt — its two children (may be leaves or subtrees)
  //   ls,  rs  — the sizes (number of leaves) under lft and rgt
  //   lb_      — is lft large enough to form its own cluster? (ls >= min_pts)
  //   rb_      — same for rgt
  for (int m = nm - 1; m >= 0; m--) {
    const int    nid = n + m;
    const int    cl  = nc[nid];
    if (cl < 0) continue;   // this node was never assigned to a cluster (skip)

    const double lam = sl[m].lambda;    // density at which this merge occurs
    const double lb  = nb[nid];         // density at which the current cluster was born
    const int    lft = sl[m].left,  ls = nd_sz(lft, sl, n);
    const int    rgt = sl[m].right, rs = nd_sz(rgt, sl, n);
    const bool   lb_ = ls >= min_pts;   // left child is large enough
    const bool   rb_ = rs >= min_pts;   // right child is large enough

    if (lb_ && rb_) {
      // ── TRUE SPLIT ────────────────────────────────────────────────────────
      // Both sides are large enough to be independent clusters.
      // All ls+rs points in the current cluster exit it at lambda=lam
      // (some will continue into child clusters, but from cl's perspective
      // they leave at this density level).
      // Stability contribution: each point has lived in cl from lb to lam.
      cl_stab[cl] += (double)(ls + rs) * (lam - lb);
      // Create two new child clusters, both born at lam.
      int cl_l = new_cl(cl, lam), cl_r = new_cl(cl, lam);
      nc[lft] = cl_l; nb[lft] = lam;
      nc[rgt] = cl_r; nb[rgt] = lam;

    } else if (lb_) {
      // ── RIGHT FALLS OUT AS NOISE ──────────────────────────────────────────
      // The right child is too small (rs < min_pts) to form a cluster on its
      // own — it dissolves into noise at this density level.
      // Stability: the rs points that fall out contributed (lam - lb) each.
      cl_stab[cl] += (double)rs * (lam - lb);
      // The left child is large enough → the cluster continues through it.
      nc[lft] = cl;   nb[lft] = lb;
      // The right child and its subtree get no cluster (nc stays -1).
      nc[rgt] = -1;

    } else if (rb_) {
      // ── LEFT FALLS OUT AS NOISE ───────────────────────────────────────────
      // Mirror of the previous case.
      cl_stab[cl] += (double)ls * (lam - lb);
      nc[lft] = -1;
      nc[rgt] = cl;   nb[rgt] = lb;

    } else {
      // ── BOTH SIDES TOO SMALL: CLUSTER DISSOLVES ───────────────────────────
      // Neither child meets the min_pts threshold.  The cluster cannot survive
      // this split; all its remaining points exit as noise at lam.
      cl_stab[cl] += (double)(ls + rs) * (lam - lb);
      // Mark both children as "no cluster" so the loop skips their subtrees.
      nc[lft] = -1;
      nc[rgt] = -1;
    }
  }

  const int ncl = next_cl;
  if (ncl == 0)   // no clusters were ever created (all noise)
    return List::create(Named("labels") = IntegerVector(n, 0),
                        Named("n_mst_edges") = nm);

  // ── Step 5: EOM cluster extraction ───────────────────────────────────────
  //
  // We now have a condensed tree of candidate clusters, each with a stability.
  // EOM selects a non-overlapping subset that maximises total stability.
  //
  // BOTTOM-UP PASS (process clusters from highest ID to lowest = leaves first):
  // For a leaf cluster (no children): sel_stab = its own stability.
  // For an internal cluster: sel_stab = max(its own stability, sum of children's).
  // "sel_self" records whether THIS cluster preferred its own stability.
  vector<double> sel_stab(ncl);
  vector<bool>   sel_self(ncl, false);
  for (int i = 0; i < ncl; i++) sel_stab[i] = cl_stab[i];

  for (int cl = ncl - 1; cl >= 0; cl--) {
    if (cl_ch[cl].empty()) {
      // Leaf cluster: always select itself (no children to compare against).
      sel_self[cl] = true;
    } else {
      double csum = 0.0;
      for (int ch : cl_ch[cl]) csum += sel_stab[ch];
      if (cl_stab[cl] >= csum) {
        // This cluster's own stability beats what its children offer combined.
        // → Select this cluster (merge children back into parent).
        sel_self[cl]  = true;
        sel_stab[cl]  = cl_stab[cl];
      } else {
        // Children collectively offer more stability → prefer them.
        sel_self[cl]  = false;
        sel_stab[cl]  = csum;    // propagate children's total upward
      }
    }
  }

  // TOP-DOWN PASS (ascending ID = root before children):
  // Propagate an "active" flag.  A cluster is SELECTED iff:
  //   (1) it is active (reachable from the root without passing a selected ancestor), AND
  //   (2) sel_self[cl] is true.
  // When a cluster is selected, we do NOT activate its children — they are
  // subsumed by the selected ancestor.
  vector<bool> active(ncl, false), selected(ncl, false);
  active[0] = true;   // the root cluster is always active

  for (int cl = 0; cl < ncl; cl++) {
    if (!active[cl]) continue;
    if (sel_self[cl]) {
      selected[cl] = true;
      // Do NOT activate children; they are covered by this cluster.
    } else {
      // This cluster deferred to its children → activate them.
      for (int ch : cl_ch[cl]) active[ch] = true;
    }
  }

  // ── Step 6: Map each cluster to its lowest selected ancestor ──────────────
  // rep[cl] = the lowest (most specific) selected cluster in cl's ancestry.
  // We process clusters in ascending ID order, so rep[parent] is guaranteed to
  // be set before we process any child.
  vector<int> rep(ncl, -1);
  for (int cl = 0; cl < ncl; cl++) {
    if (selected[cl]) {
      rep[cl] = cl;                                       // self-selected
    } else {
      int par = cl_par[cl];
      rep[cl] = (par >= 0) ? rep[par] : -1;              // inherit from parent
    }
  }

  // ── Step 7: Assign 1-indexed integer labels to selected clusters ──────────
  // Labels: 0 = noise, 1, 2, ... = cluster IDs (matching dbscan convention).
  vector<int> lmap(ncl, 0);
  int next_lbl = 1;
  for (int cl = 0; cl < ncl; cl++) if (selected[cl]) lmap[cl] = next_lbl++;

  // ── Step 8: Build SL-tree parent pointers ────────────────────────────────
  // We need to walk UPWARD from any point to find the first SL node whose
  // condensed cluster is selected.  To walk upward we need parent pointers.
  //
  // Why is a parent-walk necessary?
  //   During the condensed-tree pass (Step 4), leaf points on the SMALL SIDE
  //   of a split get nc[leaf] = -1 (no cluster).  But their PARENT SL node
  //   (the internal merge node that joined them) still has a valid nc[].
  //   Walking up one level resolves this correctly.
  //
  // Points that walk all the way to -1 (no selected ancestor) are noise.
  vector<int> sl_par(n + nm, -1);
  for (int m = 0; m < nm; m++) {
    int nid = n + m;
    sl_par[sl[m].left]  = nid;
    sl_par[sl[m].right] = nid;
  }

  // ── Step 9: Label each point ──────────────────────────────────────────────
  IntegerVector labels(n, 0);   // default: noise
  for (int i = 0; i < n; i++) {
    int node = i;
    // Walk up the SL tree until we find a node with a valid condensed cluster.
    while (node != -1) {
      int cl = nc[node];
      if (cl >= 0) {
        // Found a node whose cluster is assigned.  Look up its selected representative.
        int r = rep[cl];
        if (r >= 0) labels[i] = lmap[r];   // r = -1 means not in any selected cluster → noise
        break;
      }
      node = sl_par[node];   // no cluster here → go up to the parent merge node
    }
  }

  return List::create(Named("labels") = labels, Named("n_mst_edges") = nm);
}
