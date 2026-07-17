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
#include <queue>
#include <utility>
#include <vector>
#include <limits>

using namespace Rcpp;
using std::greater;
using std::pair;
using std::priority_queue;
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

// =============================================================================
// KD-tree Borůvka MST  (no pre-built kNN matrix — equivalent to Python hdbscan)
// =============================================================================
//
// This function takes the raw data matrix X directly and builds a single KD-tree
// (via nanoflann) that is queried on-demand during each Borůvka round.
//
// The key difference from hdbscan_boruvka_cpp():
//   - No fixed k parameter for the user.
//   - The KD-tree is built once (O(n log n)).
//   - In each Borůvka round, each point queries k_round neighbors.
//     k_round starts at max(min_pts, 3) and doubles whenever a round makes no
//     progress — matching the adaptive, connectivity-guaranteed behaviour of
//     Python hdbscan's boruvka_kdtree algorithm.
//   - Because neighbors are returned in Euclidean order and mrd >= eucl,
//     we can prune the per-point scan early: once eucl exceeds the current
//     best mrd for the component, no further neighbor can improve it.
//
// Memory: O(n log n) for the KD-tree + O(n * k_round) for one round's queries.
// No O(n * k_max) matrix is ever materialised.

#ifdef _OPENMP
#include <omp.h>
#endif
#include "nanoflann.hpp"

// Adaptor: lets nanoflann index an R NumericMatrix (column-major storage).
struct RMatAdaptor {
  const double* data;   // pointer to matrix data (column-major)
  int           n, d;

  RMatAdaptor(const NumericMatrix& X)
    : data(X.begin()), n(X.nrow()), d(X.ncol()) {}

  // Required by nanoflann.
  inline size_t kdtree_get_point_count() const { return (size_t)n; }

  // Return the dim-th coordinate of point idx.
  // Column-major: element (row, col) is at data[row + col * nrow].
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    return data[(int)idx + (int)dim * n];
  }

  // No bounding-box hint — let nanoflann compute it.
  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const { return false; }
};

// Convenience typedef for a dynamic-dimension L2 KD-tree.
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, RMatAdaptor>,
    RMatAdaptor,
    -1   // -1 = compile-time-unknown dimension (dynamic)
> KDTree_t;

// =============================================================================
// Dual-tree Borůvka MST
// =============================================================================
//
// A single-tree Borůvka query asks the KD-tree for the k nearest neighbours
// of each point, where k must grow across rounds as components merge.  Cost
// per round is O(n × k × log n).
//
// Dual-tree Borůvka (Curtin et al. 2013) traverses two copies of the same
// KD-tree simultaneously and prunes entire pairs of subtrees when the MRD
// lower bound between their bounding boxes already exceeds the current best
// outgoing edge for every component in both subtrees.  Per-round work drops
// to O(n log n); the k-growing heuristic is eliminated entirely.
//
// Post-build annotation
// ---------------------
// nanoflann stores split values per node but not bounding boxes.  A one-time
// DFS computes tight AABBs for every node (O(n × d)) and stores them in flat
// arrays indexed by pre-order node ID.
//
// Per-round state (refreshed O(n) per round)
// -------------------------------------------
//   max_bw[node]      max bw[root[i]] over all points i in subtree.
//                     Pruning condition: mrd_lb >= max(max_bw[qa], max_bw[ra]).
//
//   single_comp[node] Unique component root if every point in subtree belongs
//                     to the same component; −1 otherwise.  Prunes subtree
//                     pairs where no cross-component edge exists.
//
// Carry-forward
// -------------
// bw[ri] is NOT reset between rounds.  Valid entries provide tight max_bw
// values from the first traversal step, increasing subtree pruning.

// ── Per-node annotation (built once after tree.buildIndex()) ─────────────────
struct KDAnnotation {
    int n_nodes, d, n;
    const double*    data;   // raw column-major pointer into the R matrix
    vector<double>   data_rm; // row-major copy: data_rm[i*d + dim] — cache-friendly inner loop
    vector<uint32_t> vAcc;   // copy of tree.vAcc_ (sort-pos → point index)

    // Static (built once)
    vector<int>    lc, rc;          // child node IDs; −1 = leaf
    vector<int>    pt_beg, pt_end;  // vAcc range for leaf nodes
    vector<int>    sz;              // subtree point count
    vector<double> lo, hi;          // AABB: index as [node_id * d + dim]
    vector<double> min_core;        // min core distance in subtree

    // Per-round (refreshed by dt_round_update each round)
    vector<double> max_bw;          // max bw[root[i]] in subtree
    vector<int>    single_comp;     // uniform component root, or −1
};

static int count_kd_nodes(const KDTree_t::Node* nd)
{
    if (!nd->child1) return 1;
    return 1 + count_kd_nodes(nd->child1) + count_kd_nodes(nd->child2);
}

// Recursive post-build fill: pre-order node IDs, bottom-up bboxes.
static void annotate_kd(
    const KDTree_t::Node* nd,
    const KDTree_t&       tree,
    const vector<double>& core,
    KDAnnotation&         ann,
    int&                  nid)
{
    const int my = nid++;
    const int d  = ann.d, n = ann.n;
    const double* data = ann.data;
    const double inf   = std::numeric_limits<double>::infinity();

    if (!nd->child1) {
        // ── Leaf ─────────────────────────────────────────────────────────────
        const int beg = (int)nd->node_type.lr.left;
        const int end = (int)nd->node_type.lr.right;
        ann.lc[my] = ann.rc[my] = -1;
        ann.pt_beg[my] = beg;  ann.pt_end[my] = end;
        ann.sz[my]     = end - beg;

        double mc = inf;
        for (int dim = 0; dim < d; dim++) {
            ann.lo[my * d + dim] =  inf;
            ann.hi[my * d + dim] = -inf;
        }
        for (int k = beg; k < end; k++) {
            const int pt = (int)tree.vAcc_[k];
            for (int dim = 0; dim < d; dim++) {
                const double v = data[pt + dim * n];
                if (v < ann.lo[my * d + dim]) ann.lo[my * d + dim] = v;
                if (v > ann.hi[my * d + dim]) ann.hi[my * d + dim] = v;
            }
            if (core[pt] < mc) mc = core[pt];
        }
        ann.min_core[my] = mc;
    } else {
        // ── Internal ─────────────────────────────────────────────────────────
        const int lc_id = nid;
        annotate_kd(nd->child1, tree, core, ann, nid);
        const int rc_id = nid;
        annotate_kd(nd->child2, tree, core, ann, nid);

        ann.lc[my] = lc_id;  ann.rc[my] = rc_id;
        ann.sz[my] = ann.sz[lc_id] + ann.sz[rc_id];
        ann.min_core[my] = std::min(ann.min_core[lc_id], ann.min_core[rc_id]);
        ann.pt_beg[my] = ann.pt_end[my] = -1;
        for (int dim = 0; dim < d; dim++) {
            ann.lo[my * d + dim] = std::min(ann.lo[lc_id * d + dim],
                                            ann.lo[rc_id * d + dim]);
            ann.hi[my * d + dim] = std::max(ann.hi[lc_id * d + dim],
                                            ann.hi[rc_id * d + dim]);
        }
    }
    ann.max_bw[my]       = std::numeric_limits<double>::infinity();
    ann.single_comp[my]  = -1;
}

static KDAnnotation build_kd_annotation(
    const KDTree_t&       tree,
    const vector<double>& core,
    const double*         data, int n, int d)
{
    const int nn = count_kd_nodes(tree.root_node_);
    KDAnnotation ann;
    ann.n_nodes = nn;  ann.d = d;  ann.n = n;  ann.data = data;
    ann.vAcc.assign(tree.vAcc_.begin(), tree.vAcc_.end());

    // Row-major copy: transpose column-major R matrix so inner loop stride = 1.
    ann.data_rm.resize((size_t)n * d);
    for (int dim = 0; dim < d; dim++)
        for (int i = 0; i < n; i++)
            ann.data_rm[(size_t)i * d + dim] = data[i + (size_t)dim * n];

    ann.lc.resize(nn);  ann.rc.resize(nn);
    ann.pt_beg.resize(nn);  ann.pt_end.resize(nn);
    ann.sz.resize(nn);
    ann.lo.resize((size_t)nn * d);
    ann.hi.resize((size_t)nn * d);
    ann.min_core.resize(nn);
    ann.max_bw.resize(nn, std::numeric_limits<double>::infinity());
    ann.single_comp.resize(nn, -1);

    int nid = 0;
    annotate_kd(tree.root_node_, tree, core, ann, nid);
    return ann;
}

// Per-round: recompute max_bw and single_comp bottom-up from bw[] / roots[].
static void dt_round_update(
    KDAnnotation&         ann,
    const vector<double>& bw,
    const vector<int>&    roots,
    int                   node_id)
{
    if (ann.lc[node_id] < 0) {
        // Leaf: scan all points
        double mx   = 0.0;
        int    comp = -2;  // −2 = "not yet seen"
        for (int k = ann.pt_beg[node_id]; k < ann.pt_end[node_id]; k++) {
            const int pt = (int)ann.vAcc[k];
            const int ri = roots[pt];
            const double w = bw[ri];
            if (w > mx) mx = w;
            if (comp == -2) comp = ri;
            else if (comp != ri) comp = -1;
        }
        ann.max_bw[node_id]      = mx;
        ann.single_comp[node_id] = (comp == -2) ? -1 : comp;
    } else {
        const int lc = ann.lc[node_id], rc = ann.rc[node_id];
        dt_round_update(ann, bw, roots, lc);
        dt_round_update(ann, bw, roots, rc);
        ann.max_bw[node_id] = std::max(ann.max_bw[lc], ann.max_bw[rc]);
        const int lcomp = ann.single_comp[lc], rcomp = ann.single_comp[rc];
        ann.single_comp[node_id] =
            (lcomp >= 0 && lcomp == rcomp) ? lcomp : -1;
    }
}

// Recursive dual-tree step.
// Updates bw[ri] / bu[ri] / bv[ri] for components that can be improved by
// edges between subtree qa and subtree ra.
static void dt_boruvka(
    int                    qa,
    int                    ra,
    const KDAnnotation&    ann,
    const vector<double>&  core,
    const vector<int>&     roots,
    vector<double>&        bw,
    vector<int>&           bu,
    vector<int>&           bv)
{
    // ── Same-component pruning ────────────────────────────────────────────────
    // If all points in both subtrees belong to the same component there are no
    // cross-component edges to find.
    const int qc = ann.single_comp[qa], rc_ = ann.single_comp[ra];
    if (qc >= 0 && qc == rc_) return;

    // ── Euclidean lower bound between bounding boxes ──────────────────────────
    const int d = ann.d;
    double eucl_lb_sq = 0.0;
    for (int dim = 0; dim < d; dim++) {
        const double gap = std::max(0.0,
            std::max(ann.lo[qa * d + dim] - ann.hi[ra * d + dim],
                     ann.lo[ra * d + dim] - ann.hi[qa * d + dim]));
        eucl_lb_sq += gap * gap;
    }
    // MRD lower bound: any edge between qa and ra has MRD ≥ mrd_lb.
    const double mrd_lb = std::max(
        {ann.min_core[qa], ann.min_core[ra], std::sqrt(eucl_lb_sq)});

    // ── Distance pruning ──────────────────────────────────────────────────────
    // Prune if even the weakest component in either subtree already has a
    // recorded outgoing edge at least as good as the best possible here.
    if (mrd_lb >= std::max(ann.max_bw[qa], ann.max_bw[ra])) return;

    const bool qa_leaf = (ann.lc[qa] < 0);
    const bool ra_leaf = (ann.lc[ra] < 0);

    // ── Leaf × leaf: brute-force ──────────────────────────────────────────────
    if (qa_leaf && ra_leaf) {
        const int beg_a = ann.pt_beg[qa], end_a = ann.pt_end[qa];
        const int beg_b = ann.pt_beg[ra], end_b = ann.pt_end[ra];
        const bool self_leaf = (qa == ra);
        // Use row-major copy: data_rm[i*d + dim] so all d dims of point i
        // are contiguous — avoids n×8-byte column-major strides in inner loop.
        const double* rm = ann.data_rm.data();

        for (int ki = beg_a; ki < end_a; ki++) {
            const int i  = (int)ann.vAcc[ki];
            const int ri = roots[i];
            const double* pi = rm + (size_t)i * d;
            // For self-comparison, only upper triangle to avoid duplicate pairs.
            const int kj0 = self_leaf ? ki + 1 : beg_b;
            for (int kj = kj0; kj < end_b; kj++) {
                const int j  = (int)ann.vAcc[kj];
                const int rj = roots[j];
                if (ri == rj) continue;

                const double* pj = rm + (size_t)j * d;
                double eucl_sq = 0.0;
                for (int dim = 0; dim < d; dim++) {
                    const double diff = pi[dim] - pj[dim];
                    eucl_sq += diff * diff;
                }
                const double eucl = std::sqrt(eucl_sq);
                const double mrd  = std::max({core[i], core[j], eucl});

                // Update best outgoing edge for both components.
                if (mrd < bw[ri]) { bw[ri] = mrd; bu[ri] = i; bv[ri] = j; }
                if (mrd < bw[rj]) { bw[rj] = mrd; bu[rj] = j; bv[rj] = i; }
            }
        }
        return;
    }

    // ── Recurse ───────────────────────────────────────────────────────────────
    if (qa == ra) {
        // Self-comparison: left-left, right-right, left-right.
        const int lc = ann.lc[qa], rc = ann.rc[qa];
        dt_boruvka(lc, lc, ann, core, roots, bw, bu, bv);
        dt_boruvka(rc, rc, ann, core, roots, bw, bu, bv);
        dt_boruvka(lc, rc, ann, core, roots, bw, bu, bv);
        return;
    }
    // Cross-comparison: split the larger subtree.
    if (qa_leaf) {
        dt_boruvka(qa, ann.lc[ra], ann, core, roots, bw, bu, bv);
        dt_boruvka(qa, ann.rc[ra], ann, core, roots, bw, bu, bv);
    } else if (ra_leaf || ann.sz[qa] >= ann.sz[ra]) {
        dt_boruvka(ann.lc[qa], ra, ann, core, roots, bw, bu, bv);
        dt_boruvka(ann.rc[qa], ra, ann, core, roots, bw, bu, bv);
    } else {
        dt_boruvka(qa, ann.lc[ra], ann, core, roots, bw, bu, bv);
        dt_boruvka(qa, ann.rc[ra], ann, core, roots, bw, bu, bv);
    }
}

static vector<Edge> boruvka_mst_kd(
    const KDTree_t&       tree,
    const RMatAdaptor&    adaptor,
    const vector<double>& core,
    int n, int d, int min_pts)
{
    vector<Edge> mst;  mst.reserve(n - 1);
    UF uf(n);
    const double dbl_inf = std::numeric_limits<double>::infinity();

    // ── Build per-node annotation once ───────────────────────────────────────
    KDAnnotation ann = build_kd_annotation(tree, core, adaptor.data, n, d);

    // bw[ri] = best outgoing edge weight for component ri; carried forward.
    vector<double> bw(n, dbl_inf);
    vector<int>    bu(n, -1), bv(n, -1);
    vector<int>    roots(n);

    // ── Warm-up: seed bw[] from kNN so round 1 can prune ─────────────────────
    // Without this, bw[]=∞ → max_bw[every_node]=∞ → zero pruning in round 1 →
    // O(n²) leaf-pair work.  Re-running knnSearch here costs the same as the
    // core-distance pass (already paid in hdbscan_kdtree_cpp) but converts
    // round 1 from O(n²) to O(n log n) by giving every singleton a finite bw.
    {
        const int kc = std::min(min_pts + 1, n);
        vector<uint32_t> idx(kc);
        vector<double>   dsq(kc);
        vector<double>   qpt(d);
        for (int i = 0; i < n; i++) {
            for (int dim = 0; dim < d; dim++) qpt[dim] = adaptor.data[i + dim * n];
            tree.knnSearch(qpt.data(), (size_t)kc, idx.data(), dsq.data());
            for (int t = 0; t < kc; t++) {
                const int    j    = (int)idx[t];
                if (j == i) continue;
                const double eucl = std::sqrt(dsq[t]);
                const double mrd  = std::max({core[i], core[j], eucl});
                // Update bw for both singleton components (round 1: root[x]=x).
                if (mrd < bw[i]) { bw[i] = mrd; bu[i] = i; bv[i] = j; }
                if (mrd < bw[j]) { bw[j] = mrd; bu[j] = j; bv[j] = i; }
            }
        }
    }

    while ((int)mst.size() < n - 1 && uf.nc > 1) {

        // Pre-compute union-find roots (path-compressed, serial).
        for (int i = 0; i < n; i++) roots[i] = uf.find(i);

        // Refresh per-node max_bw and single_comp from current bw[]/roots[].
        dt_round_update(ann, bw, roots, 0);

        // ── Phase A: dual-tree search ─────────────────────────────────────────
        dt_boruvka(0, 0, ann, core, roots, bw, bu, bv);

        // ── Phase B: add best edges, merge components ─────────────────────────
        bool any = false;
        for (int r = 0; r < n; r++) {
            if (bu[r] < 0) continue;
            if (uf.unite(bu[r], bv[r]) >= 0) {
                mst.push_back({bw[r], bu[r], bv[r]});
                any = true;
                if ((int)mst.size() == n - 1) break;
            }
        }
        if (!any) break;

        // ── Carry-forward cleanup ─────────────────────────────────────────────
        // Clear bw entries whose root was subsumed or whose edge is now
        // intra-component; valid entries carry over to the next round.
        for (int r = 0; r < n; r++) {
            if (bu[r] < 0) continue;
            if (uf.find(r) != r || uf.find(bu[r]) == uf.find(bv[r]))
                { bw[r] = dbl_inf; bu[r] = -1; bv[r] = -1; }
        }
    }

    std::sort(mst.begin(), mst.end(),
              [](const Edge& a, const Edge& b){ return a.w < b.w; });
    return mst;
}

// ── Shared postprocessing: MST → single-linkage → condensed tree → EOM ───────
static List mst_to_hdbscan_result(const vector<Edge>& mst, int n, int min_pts)
{
  const int nm = (int)mst.size();
  if (nm == 0)
    return List::create(Named("labels") = IntegerVector(n, 0),
                        Named("n_mst_edges") = 0);

  vector<SLNode> sl = build_sl(mst, n);

  const int ntot = n + nm;
  vector<int>    nc_arr(ntot, -1);
  vector<double> nb_arr(ntot, 0.0);

  vector<double>      cl_stab;
  vector<int>         cl_par;
  vector<vector<int>> cl_ch;
  int next_cl = 0;

  auto new_cl = [&](int par, double) -> int {
    int id = next_cl++;
    cl_stab.push_back(0.0);
    cl_par.push_back(par);
    cl_ch.push_back(vector<int>());
    if (par >= 0) cl_ch[par].push_back(id);
    return id;
  };

  nc_arr[n + nm - 1] = new_cl(-1, 0.0);
  nb_arr[n + nm - 1] = 0.0;

  for (int m = nm - 1; m >= 0; m--) {
    const int    nid = n + m;
    const int    cl  = nc_arr[nid];
    if (cl < 0) continue;
    const double lam = sl[m].lambda;
    const double lb  = nb_arr[nid];
    const int    lft = sl[m].left,  ls = nd_sz(lft, sl, n);
    const int    rgt = sl[m].right, rs = nd_sz(rgt, sl, n);
    const bool   lb_ = ls >= min_pts, rb_ = rs >= min_pts;
    if (lb_ && rb_) {
      cl_stab[cl] += (double)(ls + rs) * (lam - lb);
      int cl_l = new_cl(cl, lam), cl_r = new_cl(cl, lam);
      nc_arr[lft] = cl_l; nb_arr[lft] = lam;
      nc_arr[rgt] = cl_r; nb_arr[rgt] = lam;
    } else if (lb_) {
      cl_stab[cl] += (double)rs * (lam - lb);
      nc_arr[lft] = cl; nb_arr[lft] = lb; nc_arr[rgt] = -1;
    } else if (rb_) {
      cl_stab[cl] += (double)ls * (lam - lb);
      nc_arr[lft] = -1; nc_arr[rgt] = cl; nb_arr[rgt] = lb;
    } else {
      cl_stab[cl] += (double)(ls + rs) * (lam - lb);
      nc_arr[lft] = -1; nc_arr[rgt] = -1;
    }
  }

  const int ncl = next_cl;
  if (ncl == 0)
    return List::create(Named("labels") = IntegerVector(n, 0),
                        Named("n_mst_edges") = nm);

  vector<double> sel_stab(ncl);
  vector<bool>   sel_self(ncl, false);
  for (int i = 0; i < ncl; i++) sel_stab[i] = cl_stab[i];
  for (int cl = ncl - 1; cl >= 0; cl--) {
    if (cl_ch[cl].empty()) {
      sel_self[cl] = true;
    } else {
      double csum = 0.0;
      for (int ch : cl_ch[cl]) csum += sel_stab[ch];
      if (cl_stab[cl] >= csum) { sel_self[cl] = true;  sel_stab[cl] = cl_stab[cl]; }
      else                     { sel_self[cl] = false; sel_stab[cl] = csum; }
    }
  }

  vector<bool> active(ncl, false), selected(ncl, false);
  active[0] = true;
  for (int cl = 0; cl < ncl; cl++) {
    if (!active[cl]) continue;
    if (sel_self[cl]) selected[cl] = true;
    else for (int ch : cl_ch[cl]) active[ch] = true;
  }

  vector<int> rep(ncl, -1);
  for (int cl = 0; cl < ncl; cl++) {
    if (selected[cl]) rep[cl] = cl;
    else { int par = cl_par[cl]; rep[cl] = (par >= 0) ? rep[par] : -1; }
  }

  vector<int> lmap(ncl, 0);
  int next_lbl = 1;
  for (int cl = 0; cl < ncl; cl++) if (selected[cl]) lmap[cl] = next_lbl++;

  vector<int> sl_par(n + nm, -1);
  for (int m = 0; m < nm; m++) {
    sl_par[sl[m].left]  = n + m;
    sl_par[sl[m].right] = n + m;
  }

  IntegerVector labels(n, 0);
  for (int i = 0; i < n; i++) {
    int node = i;
    while (node != -1) {
      int cl = nc_arr[node];
      if (cl >= 0) { int r = rep[cl]; if (r >= 0) labels[i] = lmap[r]; break; }
      node = sl_par[node];
    }
  }
  return List::create(Named("labels") = labels, Named("n_mst_edges") = nm);
}

//' HDBSCAN via Boruvka MST with KD-tree queries (no fixed k)
//'
//' Internal function called by \code{cluster_docs.hdbscan_clustering()} when
//' \code{knn = "kdtree"}.  Unlike \code{hdbscan_boruvka_cpp()}, this function
//' takes the raw data matrix \code{X} and builds its own KD-tree (nanoflann),
//' querying it on-demand during each Boruvka round.  No k parameter is exposed:
//' the search radius grows automatically until the MST is fully connected.
//' This replicates the behaviour of Python hdbscan's \code{boruvka_kdtree}
//' algorithm.
//'
//' @param X Numeric matrix (n x d) of point coordinates (e.g. 5-D UMAP embedding).
//' @param min_pts Integer minimum cluster size / core-distance order.
//' @return Named list: \code{labels} (IntegerVector, 0 = noise),
//'   \code{n_mst_edges} (int, always n-1 when data is connected).
//' @keywords internal
// [[Rcpp::export]]
List hdbscan_kdtree_cpp(NumericMatrix X, int min_pts)
{
  const int n = X.nrow();
  const int d = X.ncol();

  // ── Step 1: Build KD-tree (once) ─────────────────────────────────────────
  RMatAdaptor adaptor(X);
  KDTree_t tree(d, adaptor,
                nanoflann::KDTreeSingleIndexAdaptorParams(/*leaf_max_size=*/10));
  tree.buildIndex();

  // ── Step 2: Core distances via KD-tree ───────────────────────────────────
  // core[i] = distance to i's min_pts-th nearest neighbor (excluding i itself).
  // Query min_pts+1 neighbors; the first returned is i itself (dist 0).
  {
    // Sanity: if min_pts >= n, clamp.
  }
  int kc = std::min(min_pts + 1, n);
  vector<double> core(n);
  {
    vector<uint32_t> idx(kc);
    vector<double>   dsq(kc);
    vector<double> qpt(d);
    for (int i = 0; i < n; i++) {
      for (int dim = 0; dim < d; dim++) qpt[dim] = adaptor.data[i + dim * n];
      tree.knnSearch(qpt.data(), (size_t)kc, idx.data(), dsq.data());
      // kc results: idx[0] = i (dist 0), idx[kc-1] = min_pts-th neighbor.
      core[i] = std::sqrt(dsq[kc - 1]);
    }
  }

  // ── Step 3: Borůvka MST ──────────────────────────────────────────────────
  vector<Edge> mst = boruvka_mst_kd(tree, adaptor, core, n, d, min_pts);

  // ── Steps 4–9: MST → labels ──────────────────────────────────────────────
  return mst_to_hdbscan_result(mst, n, min_pts);
}

// =============================================================================
// Ball-tree dual-tree Borůvka MST
// =============================================================================
//
// Each node stores a bounding hypersphere (centroid + radius).  The Euclidean
// lower bound between balls qa and ra is:
//
//   ball_lb = max(0, dist(centroid_qa, centroid_ra) − radius_qa − radius_ra)
//
// This bound is dimension-independent: unlike axis-aligned boxes, a ball does
// not expand in irrelevant dimensions, so pruning remains effective in 5-D
// (where KD-tree AABBs overlap on ~99% of leaf pairs for random data).
//
// Build: median split on the dimension of maximum spread (same heuristic as
// the KD-tree).  Centroid and radius are computed bottom-up from raw points.
// kNN search for core distances uses a priority-queue descent with ball bounds.
// kNN results from the core-distance pass are reused as the Borůvka warm-up,
// so no extra kNN pass is needed.

struct BallAnnotation {
    int n, d, n_nodes;
    vector<double> data_rm;    // row-major copy: [i*d + dim]
    vector<int>    vAcc;       // point ordering built during tree construction

    // Static (built once)
    vector<int>    lc, rc;
    vector<int>    pt_beg, pt_end;
    vector<int>    sz;
    vector<double> centroid;   // [node_id * d + dim]
    vector<double> radius;     // [node_id]
    vector<double> min_core;   // min core distance in subtree

    // Per-round (refreshed each Borůvka round)
    vector<double> max_bw;
    vector<int>    single_comp;
};

// Recursive build: reorders pts[beg..end) and fills annotation arrays.
static void bt_build_rec(
    BallAnnotation& ann,
    vector<int>&    pts,      // working index array, reordered in place
    int beg, int end,
    int& nid, int leaf_max)
{
    const int my = nid++;
    const int sz = end - beg;
    const int d  = ann.d;
    const double* rm = ann.data_rm.data();

    // Centroid = mean of all points in this subtree
    double* c = ann.centroid.data() + (size_t)my * d;
    for (int dim = 0; dim < d; dim++) c[dim] = 0.0;
    for (int k = beg; k < end; k++) {
        const double* p = rm + (size_t)pts[k] * d;
        for (int dim = 0; dim < d; dim++) c[dim] += p[dim];
    }
    for (int dim = 0; dim < d; dim++) c[dim] /= sz;

    // Radius = max dist from centroid to any point in subtree
    double r_sq = 0.0;
    for (int k = beg; k < end; k++) {
        const double* p = rm + (size_t)pts[k] * d;
        double dsq = 0.0;
        for (int dim = 0; dim < d; dim++) { double dv = p[dim] - c[dim]; dsq += dv*dv; }
        if (dsq > r_sq) r_sq = dsq;
    }
    ann.radius[my]      = std::sqrt(r_sq);
    ann.sz[my]          = sz;
    ann.min_core[my]    = std::numeric_limits<double>::infinity();
    ann.max_bw[my]      = std::numeric_limits<double>::infinity();
    ann.single_comp[my] = -1;

    if (sz <= leaf_max) {
        ann.lc[my] = ann.rc[my] = -1;
        ann.pt_beg[my] = beg;
        ann.pt_end[my] = end;
        for (int k = beg; k < end; k++) ann.vAcc[k] = pts[k];
        return;
    }
    ann.pt_beg[my] = ann.pt_end[my] = -1;

    // Split dimension = max coordinate spread
    int    split_dim = 0;
    double max_sp    = -1.0;
    for (int dim = 0; dim < d; dim++) {
        double lo =  1e300, hi = -1e300;
        for (int k = beg; k < end; k++) {
            double v = rm[(size_t)pts[k] * d + dim];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        double sp = hi - lo;
        if (sp > max_sp) { max_sp = sp; split_dim = dim; }
    }

    // Median partition via nth_element
    int mid = beg + sz / 2;
    const double* rm2 = rm;
    const int d2 = d, sd = split_dim;
    std::nth_element(pts.begin() + beg, pts.begin() + mid, pts.begin() + end,
        [rm2, d2, sd](int a, int b){
            return rm2[(size_t)a * d2 + sd] < rm2[(size_t)b * d2 + sd];
        });

    const int lc_id = nid;
    bt_build_rec(ann, pts, beg, mid, nid, leaf_max);
    const int rc_id = nid;
    bt_build_rec(ann, pts, mid, end, nid, leaf_max);
    ann.lc[my] = lc_id;
    ann.rc[my] = rc_id;
}

static BallAnnotation build_ball_annotation(
    const double* data, int n, int d, int leaf_max = 10)
{
    BallAnnotation ann;
    ann.n = n; ann.d = d;

    // Row-major copy
    ann.data_rm.resize((size_t)n * d);
    for (int dim = 0; dim < d; dim++)
        for (int i = 0; i < n; i++)
            ann.data_rm[(size_t)i * d + dim] = data[i + (size_t)dim * n];

    // Safe upper bound on node count.  Median-split trees with n points and
    // leaf_max per leaf have at most 2*ceil(n/leaf_max) leaves (since
    // 2^ceil(log2(x)) ≤ 2x), giving total nodes ≤ 4*n/leaf_max + 4.
    const int max_nodes = 4 * n / leaf_max + 20;
    ann.n_nodes = max_nodes;
    ann.lc.resize(max_nodes);      ann.rc.resize(max_nodes);
    ann.pt_beg.resize(max_nodes);  ann.pt_end.resize(max_nodes);
    ann.sz.resize(max_nodes);
    ann.centroid.resize((size_t)max_nodes * d);
    ann.radius.resize(max_nodes);
    ann.min_core.resize(max_nodes,    std::numeric_limits<double>::infinity());
    ann.max_bw.resize(max_nodes,      std::numeric_limits<double>::infinity());
    ann.single_comp.resize(max_nodes, -1);
    ann.vAcc.resize(n);

    vector<int> pts(n);
    std::iota(pts.begin(), pts.end(), 0);
    int nid = 0;
    bt_build_rec(ann, pts, 0, n, nid, leaf_max);
    ann.n_nodes = nid;
    return ann;
}

// Fill min_core bottom-up after core distances are known.
static void bt_fill_min_core(BallAnnotation& ann, const vector<double>& core, int nid)
{
    if (ann.lc[nid] < 0) {
        double mc = std::numeric_limits<double>::infinity();
        for (int k = ann.pt_beg[nid]; k < ann.pt_end[nid]; k++)
            mc = std::min(mc, core[ann.vAcc[k]]);
        ann.min_core[nid] = mc;
    } else {
        const int lc = ann.lc[nid], rc = ann.rc[nid];
        bt_fill_min_core(ann, core, lc);
        bt_fill_min_core(ann, core, rc);
        ann.min_core[nid] = std::min(ann.min_core[lc], ann.min_core[rc]);
    }
}

// kNN search via priority-queue descent.  Returns k results (incl. self at
// dist=0) in ascending distance order.
static void bt_knn_search(
    const BallAnnotation& ann,
    const double*         qpt,
    int k,
    vector<uint32_t>&     out_idx,
    vector<double>&       out_dsq)
{
    const int    d   = ann.d;
    const double inf = std::numeric_limits<double>::infinity();
    const double* rm = ann.data_rm.data();

    // Max-heap of k-best (dist_sq, idx): worst at top for O(1) pruning check.
    using KP = pair<double, int>;
    priority_queue<KP> kbest;
    for (int i = 0; i < k; i++) kbest.push({inf, -1});

    // Min-heap for tree traversal: (lb_dist, node_id), nearest ball first.
    using TP = pair<double, int>;
    priority_queue<TP, vector<TP>, greater<TP>> trav;
    trav.push({0.0, 0});

    while (!trav.empty()) {
        auto [lb, nid] = trav.top();
        trav.pop();

        if (lb * lb >= kbest.top().first) break;  // all remaining nodes pruned

        if (ann.lc[nid] < 0) {
            // Leaf: brute-force update k-best
            for (int ki = ann.pt_beg[nid]; ki < ann.pt_end[nid]; ki++) {
                const int     pt = ann.vAcc[ki];
                const double* p  = rm + (size_t)pt * d;
                double dsq = 0.0;
                for (int dim = 0; dim < d; dim++) {
                    double dv = qpt[dim] - p[dim]; dsq += dv * dv;
                }
                if (dsq < kbest.top().first) { kbest.pop(); kbest.push({dsq, pt}); }
            }
        } else {
            // Internal: push children with ball lower bounds
            for (int child : {ann.lc[nid], ann.rc[nid]}) {
                const double* cc = ann.centroid.data() + (size_t)child * d;
                double cd_sq = 0.0;
                for (int dim = 0; dim < d; dim++) {
                    double dv = qpt[dim] - cc[dim]; cd_sq += dv * dv;
                }
                const double lb_c = std::max(0.0, std::sqrt(cd_sq) - ann.radius[child]);
                if (lb_c * lb_c < kbest.top().first)
                    trav.push({lb_c, child});
            }
        }
    }

    // Extract in ascending distance order: pop max-heap (descending), fill backwards
    out_idx.resize(k); out_dsq.resize(k);
    for (int i = k - 1; i >= 0; i--) {
        out_dsq[i] = kbest.top().first;
        out_idx[i] = (uint32_t)kbest.top().second;
        kbest.pop();
    }
}

// Per-round: refresh max_bw and single_comp bottom-up.
static void bt_round_update(
    BallAnnotation&       ann,
    const vector<double>& bw,
    const vector<int>&    roots,
    int                   node_id)
{
    if (ann.lc[node_id] < 0) {
        double mx = 0.0; int comp = -2;
        for (int k = ann.pt_beg[node_id]; k < ann.pt_end[node_id]; k++) {
            const int ri = roots[ann.vAcc[k]];
            const double w = bw[ri];
            if (w > mx) mx = w;
            if (comp == -2) comp = ri; else if (comp != ri) comp = -1;
        }
        ann.max_bw[node_id]      = mx;
        ann.single_comp[node_id] = (comp == -2) ? -1 : comp;
    } else {
        const int lc = ann.lc[node_id], rc = ann.rc[node_id];
        bt_round_update(ann, bw, roots, lc);
        bt_round_update(ann, bw, roots, rc);
        ann.max_bw[node_id] = std::max(ann.max_bw[lc], ann.max_bw[rc]);
        const int lcomp = ann.single_comp[lc], rcomp = ann.single_comp[rc];
        ann.single_comp[node_id] = (lcomp >= 0 && lcomp == rcomp) ? lcomp : -1;
    }
}

// Recursive dual-tree step: ball-based lower bound replaces AABB gap.
static void bt_boruvka(
    int                    qa,
    int                    ra,
    const BallAnnotation&  ann,
    const vector<double>&  core,
    const vector<int>&     roots,
    vector<double>&        bw,
    vector<int>&           bu,
    vector<int>&           bv)
{
    // Same-component pruning
    const int qc = ann.single_comp[qa], rc_ = ann.single_comp[ra];
    if (qc >= 0 && qc == rc_) return;

    // Pruning via ball lower bound — no sqrt needed in the hot path.
    // ball_lb = max(0, dist(c_qa, c_ra) − r_qa − r_ra).
    // We prune when max(mrd_lb_min, ball_lb) ≥ max_bw, where
    // mrd_lb_min = max(min_core[qa], min_core[ra]).
    const double mrd_lb_min = std::max(ann.min_core[qa], ann.min_core[ra]);
    const double max_bw_val = std::max(ann.max_bw[qa],   ann.max_bw[ra]);
    if (mrd_lb_min >= max_bw_val) return;  // already pruned by core distances

    const int d = ann.d;
    const double* cq = ann.centroid.data() + (size_t)qa * d;
    const double* cr = ann.centroid.data() + (size_t)ra * d;
    double cd_sq = 0.0;
    for (int dim = 0; dim < d; dim++) {
        double dv = cq[dim] - cr[dim]; cd_sq += dv * dv;
    }
    const double r_sum = ann.radius[qa] + ann.radius[ra];
    // If balls don't overlap, ball_lb = sqrt(cd_sq) − r_sum > 0.
    // Prune when ball_lb ≥ max_bw_val, i.e. sqrt(cd_sq) ≥ max_bw_val + r_sum,
    // i.e. cd_sq ≥ (max_bw_val + r_sum)² — no sqrt required.
    if (cd_sq > r_sum * r_sum) {
        const double thresh = max_bw_val + r_sum;
        if (cd_sq >= thresh * thresh) return;
    }

    const bool qa_leaf = (ann.lc[qa] < 0), ra_leaf = (ann.lc[ra] < 0);

    // Leaf × leaf: brute-force (row-major data for cache efficiency)
    if (qa_leaf && ra_leaf) {
        const int  beg_a = ann.pt_beg[qa], end_a = ann.pt_end[qa];
        const int  beg_b = ann.pt_beg[ra], end_b = ann.pt_end[ra];
        const bool self_leaf = (qa == ra);
        const double* rm = ann.data_rm.data();
        for (int ki = beg_a; ki < end_a; ki++) {
            const int i  = ann.vAcc[ki], ri = roots[i];
            const double* pi = rm + (size_t)i * d;
            const int kj0 = self_leaf ? ki + 1 : beg_b;
            for (int kj = kj0; kj < end_b; kj++) {
                const int j  = ann.vAcc[kj], rj = roots[j];
                if (ri == rj) continue;
                const double* pj = rm + (size_t)j * d;
                double eucl_sq = 0.0;
                for (int dim = 0; dim < d; dim++) {
                    double dv = pi[dim] - pj[dim]; eucl_sq += dv * dv;
                }
                const double eucl = std::sqrt(eucl_sq);
                const double mrd  = std::max({core[i], core[j], eucl});
                if (mrd < bw[ri]) { bw[ri] = mrd; bu[ri] = i; bv[ri] = j; }
                if (mrd < bw[rj]) { bw[rj] = mrd; bu[rj] = j; bv[rj] = i; }
            }
        }
        return;
    }

    // Recurse: self-comparison or split the larger subtree
    if (qa == ra) {
        const int lc = ann.lc[qa], rc = ann.rc[qa];
        bt_boruvka(lc, lc, ann, core, roots, bw, bu, bv);
        bt_boruvka(rc, rc, ann, core, roots, bw, bu, bv);
        bt_boruvka(lc, rc, ann, core, roots, bw, bu, bv);
        return;
    }
    if (qa_leaf) {
        bt_boruvka(qa, ann.lc[ra], ann, core, roots, bw, bu, bv);
        bt_boruvka(qa, ann.rc[ra], ann, core, roots, bw, bu, bv);
    } else if (ra_leaf || ann.sz[qa] >= ann.sz[ra]) {
        bt_boruvka(ann.lc[qa], ra, ann, core, roots, bw, bu, bv);
        bt_boruvka(ann.rc[qa], ra, ann, core, roots, bw, bu, bv);
    } else {
        bt_boruvka(qa, ann.lc[ra], ann, core, roots, bw, bu, bv);
        bt_boruvka(qa, ann.rc[ra], ann, core, roots, bw, bu, bv);
    }
}

static vector<Edge> boruvka_mst_bt(
    BallAnnotation&         ann,
    const vector<double>&   core,
    const vector<uint32_t>& warm_idx,  // stored kNN from core-dist pass: [i*kc+t]
    const vector<double>&   warm_dsq,
    int n, int /*d*/, int kc)
{
    vector<Edge> mst;  mst.reserve(n - 1);
    UF uf(n);
    const double dbl_inf = std::numeric_limits<double>::infinity();

    vector<double> bw(n, dbl_inf);
    vector<int>    bu(n, -1), bv(n, -1);
    vector<int>    roots(n);

    // Warm-up: reuse core-dist kNN to seed finite bw values before round 1.
    // Without warm-up, bw[]=∞ → max_bw[every node]=∞ → zero pruning in round 1.
    for (int i = 0; i < n; i++) {
        const uint32_t* ri  = warm_idx.data() + (size_t)i * kc;
        const double*   rds = warm_dsq.data() + (size_t)i * kc;
        for (int t = 0; t < kc; t++) {
            const int j = (int)ri[t];
            if (j == i) continue;
            const double eucl = std::sqrt(rds[t]);
            const double mrd  = std::max({core[i], core[j], eucl});
            if (mrd < bw[i]) { bw[i] = mrd; bu[i] = i; bv[i] = j; }
            if (mrd < bw[j]) { bw[j] = mrd; bu[j] = j; bv[j] = i; }
        }
    }

    while ((int)mst.size() < n - 1 && uf.nc > 1) {
        for (int i = 0; i < n; i++) roots[i] = uf.find(i);
        bt_round_update(ann, bw, roots, 0);
        bt_boruvka(0, 0, ann, core, roots, bw, bu, bv);

        bool any = false;
        for (int r = 0; r < n; r++) {
            if (bu[r] < 0) continue;
            if (uf.unite(bu[r], bv[r]) >= 0) {
                mst.push_back({bw[r], bu[r], bv[r]});
                any = true;
                if ((int)mst.size() == n - 1) break;
            }
        }
        if (!any) break;

        for (int r = 0; r < n; r++) {
            if (bu[r] < 0) continue;
            if (uf.find(r) != r || uf.find(bu[r]) == uf.find(bv[r]))
                { bw[r] = dbl_inf; bu[r] = -1; bv[r] = -1; }
        }
    }

    std::sort(mst.begin(), mst.end(),
              [](const Edge& a, const Edge& b){ return a.w < b.w; });
    return mst;
}

//' HDBSCAN via Ball-tree dual-tree Borůvka MST
//'
//' Default internal HDBSCAN implementation (called when \code{knn = "balltree"}).
//' Builds a Ball-tree from the data matrix and runs dual-tree Borůvka.  Ball
//' bounding spheres prune more effectively than axis-aligned boxes in ≥3-D,
//' keeping Borůvka rounds close to O(n log n) even on data without strong
//' cluster separation.  kNN results from the core-distance pass are reused as
//' a Borůvka warm-up, so no extra tree traversal is needed.
//'
//' @param X Numeric matrix (n x d) of point coordinates (e.g. 5-D UMAP embedding).
//' @param min_pts Integer minimum cluster size / core-distance order.
//' @return Named list: \code{labels} (IntegerVector, 0 = noise),
//'   \code{n_mst_edges} (int, always n-1 when data is connected).
//' @keywords internal
// [[Rcpp::export]]
List hdbscan_balltree_cpp(NumericMatrix X, int min_pts)
{
  const int n = X.nrow(), d = X.ncol();
  const double* raw = REAL(X);

  // ── Step 1: Build Ball-tree ───────────────────────────────────────────────
  BallAnnotation ann = build_ball_annotation(raw, n, d, /*leaf_max=*/10);

  // ── Step 2: Core distances via Ball-tree kNN; store results for warm-up ──
  const int kc = std::min(min_pts + 1, n);
  vector<double>   core(n);
  vector<uint32_t> warm_idx((size_t)n * kc);
  vector<double>   warm_dsq((size_t)n * kc);
  {
    const double* rm = ann.data_rm.data();
    vector<uint32_t> idx_buf(kc);
    vector<double>   dsq_buf(kc);
    for (int i = 0; i < n; i++) {
      bt_knn_search(ann, rm + (size_t)i * d, kc, idx_buf, dsq_buf);
      core[i] = std::sqrt(dsq_buf[kc - 1]);  // dist to min_pts-th neighbor
      for (int t = 0; t < kc; t++) {
        warm_idx[(size_t)i * kc + t] = idx_buf[t];
        warm_dsq[(size_t)i * kc + t] = dsq_buf[t];
      }
    }
  }

  // ── Step 3: Fill min_core annotation ─────────────────────────────────────
  bt_fill_min_core(ann, core, 0);

  // ── Step 4: Borůvka MST (dual-tree with warm-up bw from Step 2) ──────────
  vector<Edge> mst = boruvka_mst_bt(ann, core, warm_idx, warm_dsq, n, d, kc);

  // ── Steps 5–10: MST → labels ─────────────────────────────────────────────
  return mst_to_hdbscan_result(mst, n, min_pts);
}

