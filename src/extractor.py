import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent

PARSED_DIR  = _PROJECT_DIR / "data" / "parsed"
DATASET_DIR = _PROJECT_DIR / "data" / "dataset"

NUM_RELATIONS = 22
TYPE_MAPPING  = {0: 0, 1: 1, 2: 2, 8: 3, 9: 4}

_PMOS_FWD_TYPES  = frozenset(range(0, 4))
_NMOS_FWD_TYPES  = frozenset(range(4, 8))
_PMOS_REV_TYPES  = frozenset(range(14, 18))
_NMOS_REV_TYPES  = frozenset(range(18, 22))

_SD_SWAP = {0: 2, 2: 0, 4: 6, 6: 4, 14: 16, 16: 14, 18: 20, 20: 18}

# A MOSFET's source and drain are a PERMUTABLE pin group -- which terminal is which is
# set by bias, not by topology -- so applying _SD_SWAP to a positive region yields the
# IDENTICAL physical circuit.  Labelling it y=0 injects mislabelled POSITIVES.
#
# Verified against this repository's own VF3 ground truth.  All 18,042 matched instances
# were canonically hashed twice, with drain/source as DISTINCT relations vs MERGED:
#     match under BOTH conventions : 17,984  (99.68%)
#     match ONLY with d/s MERGED   :     31  (0.17%)   <- VF3 permuted d/s here
#     match ONLY with d/s STRICT   :      0  (0.00%)
# Those 31 instances have a netlist drain/source assignment that DISAGREES with their
# library cell, yet VF3 matched them.  That is only possible if VF3's matcher permutes
# d/s.  Hence a d/s-swapped positive would still be matched by VF3, and y=0 contradicts
# this project's own ground-truth oracle.
#
# Scale: neg_mut = 2,752 files; make_mutation_negative builds exactly two candidates and
# random.sample picks one uniformly, so ~1,376 (~4.9% of all 27,832 negatives) are
# mislabelled.  The P/N type-swap mutation is a genuine hard negative and is KEPT.
DROP_DS_MUTATION = True

def _load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def _build_full_graph(raw: dict) -> Data:

    num_nodes      = raw['num_nodes']
    raw_node_types = raw['node_type']

    mapped = [TYPE_MAPPING.get(t, 4) for t in raw_node_types]
    x = F.one_hot(torch.tensor(mapped, dtype=torch.long), num_classes=5).float()

    edge_index = torch.tensor(raw['edge_index'], dtype=torch.long)
    edge_type  = torch.tensor(raw['edge_type'],  dtype=torch.long)

    g            = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    g.num_nodes  = num_nodes
    g.node_types = torch.tensor(mapped, dtype=torch.long)
    return g

def _extract_subgraph(full_graph: Data, center_node: int, k: int) -> Data:

    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=center_node,
        num_hops=k,
        edge_index=full_graph.edge_index,
        relabel_nodes=True,
        num_nodes=full_graph.num_nodes,
    )

    sub_x         = full_graph.x[subset]
    sub_edge_type = full_graph.edge_type[edge_mask]

    sub = Data(x=sub_x, edge_index=sub_edge_index, edge_type=sub_edge_type)
    sub.num_nodes  = subset.shape[0]
    sub.node_types = full_graph.node_types[subset]
    return sub

def make_partial_negative(subgraph: Data) -> Data | None:

    nt = subgraph.node_types
    is_transistor = (nt == 3) | (nt == 4)

    transistor_ids = is_transistor.nonzero(as_tuple=True)[0]
    if transistor_ids.numel() < 2:
        return None

    ei = subgraph.edge_index
    if ei.numel() == 0:
        return None

    degree = torch.zeros(subgraph.num_nodes, dtype=torch.long)
    degree.scatter_add_(0, ei[0], torch.ones(ei.shape[1], dtype=torch.long))
    degree.scatter_add_(0, ei[1], torch.ones(ei.shape[1], dtype=torch.long))

    center_candidate = transistor_ids[degree[transistor_ids].argmax()].item()
    removable = [t.item() for t in transistor_ids if t.item() != center_candidate]

    if not removable:
        return None

    remove_id = random.choice(removable)

    keep_mask = torch.ones(subgraph.num_nodes, dtype=torch.bool)
    keep_mask[remove_id] = False

    node_map = torch.cumsum(keep_mask.long(), dim=0) - 1

    src, dst = ei[0], ei[1]
    edge_keep = keep_mask[src] & keep_mask[dst]

    new_ei = torch.stack([
        node_map[src[edge_keep]],
        node_map[dst[edge_keep]],
    ], dim=0)
    new_et = subgraph.edge_type[edge_keep]
    new_x  = subgraph.x[keep_mask]
    new_nt = subgraph.node_types[keep_mask]

    neg          = Data(x=new_x, edge_index=new_ei, edge_type=new_et)
    neg.num_nodes  = keep_mask.sum().item()
    neg.node_types = new_nt
    return neg

def make_mutation_negative(subgraph: Data) -> list[tuple[str, Data]]:
    """Returns [(variant, Data), ...] with variant in {'pn', 'ds'}.

    'pn' swaps PMOS<->NMOS relation types    -- a genuine hard negative.
    'ds' swaps drain<->source relation types -- NOT a negative; see DROP_DS_MUTATION.

    Both variants are still constructed so that the random stream in extract_dataset is
    unchanged; the 'ds' variant is discarded at the save site instead.  That keeps every
    other negative type bit-identical to the pre-fix dataset, making this a clean ablation.
    """

    results = []

    if subgraph.edge_index.numel() == 0 or subgraph.edge_type.numel() == 0:
        return results

    et = subgraph.edge_type

    pmos_fwd = torch.tensor([t.item() in _PMOS_FWD_TYPES for t in et], dtype=torch.bool)
    nmos_fwd = torch.tensor([t.item() in _NMOS_FWD_TYPES for t in et], dtype=torch.bool)
    pmos_rev = torch.tensor([t.item() in _PMOS_REV_TYPES for t in et], dtype=torch.bool)
    nmos_rev = torch.tensor([t.item() in _NMOS_REV_TYPES for t in et], dtype=torch.bool)

    has_transistor_edges = (pmos_fwd | nmos_fwd | pmos_rev | nmos_rev).any().item()
    if has_transistor_edges:
        new_et_a = et.clone()
        new_et_a[pmos_fwd] = et[pmos_fwd] + 4
        new_et_a[nmos_fwd] = et[nmos_fwd] - 4
        new_et_a[pmos_rev] = et[pmos_rev] + 4
        new_et_a[nmos_rev] = et[nmos_rev] - 4

        neg_a            = Data(x=subgraph.x, edge_index=subgraph.edge_index,
                                edge_type=new_et_a)
        neg_a.num_nodes  = subgraph.num_nodes
        neg_a.node_types = subgraph.node_types
        results.append(("pn", neg_a))

    sd_affected = torch.tensor([t.item() in _SD_SWAP for t in et], dtype=torch.bool)

    if sd_affected.any().item():
        new_et_b = torch.tensor(
            [_SD_SWAP.get(t.item(), t.item()) for t in et], dtype=torch.long
        )

        neg_b            = Data(x=subgraph.x, edge_index=subgraph.edge_index,
                                edge_type=new_et_b)
        neg_b.num_nodes  = subgraph.num_nodes
        neg_b.node_types = subgraph.node_types
        results.append(("ds", neg_b))

    return results

import networkx as nx

def target_radius(target_graph: Data, fallback_k: int) -> int:

    G = nx.Graph()
    G.add_nodes_from(range(target_graph.num_nodes))
    ei = target_graph.edge_index
    for k in range(ei.shape[1]):
        G.add_edge(int(ei[0, k]), int(ei[1, k]))
    if G.number_of_nodes() == 0 or not nx.is_connected(G):
        return fallback_k
    try:
        return max(1, nx.radius(G))
    except nx.NetworkXError:
        return fallback_k

_MATCH_RE = re.compile(r'^\s*//\s*MATCH\s+(\S+)\s+(.*)$')

def _run_vf3(vf3_bin: Path, lib_sp: Path, raw_sp: Path, out_v: Path, timeout_s: int = 1800):

    try:
        res = subprocess.run(
            [str(vf3_bin), "-l", str(lib_sp), "-s", str(raw_sp), "-o", str(out_v)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"VF3 timed out after {timeout_s}s"
    except FileNotFoundError:
        return False, f"VF3 binary not found at {vf3_bin} (build it: cd vf3_cpp && make)"
    if res.returncode != 0:
        return False, f"VF3 exited {res.returncode}: {res.stderr.strip()[:200]}"
    return True, ""

def parse_vf3_matches(out_v: Path, json_dict: dict) -> dict:

    bare_to_id = {}
    for k, v in json_dict["id_to_node_name"].items():
        bare_to_id[v.split("/")[-1].lower()] = int(k)

    matches = {}
    with open(out_v) as f:
        for line in f:
            m = _MATCH_RE.match(line)
            if not m:
                continue
            gate = m.group(1)
            ids = frozenset(
                bare_to_id[name.lower()]
                for name in m.group(2).split()
                if name.lower() in bare_to_id
            )
            if ids:
                matches.setdefault(gate, []).append(ids)
    return matches

def _expected_count(rate: float) -> int:

    if rate <= 0:
        return 0
    base = int(rate)
    return base + (1 if random.random() < (rate - base) else 0)

def extract_dataset(
    parsed_dir:  Path,
    targets_dir: Path,
    dataset_dir: Path,
    negatives:   dict,
    max_pos_per_target,
    fallback_k:  int,
    seed:        int,
    vf3_bin:     Path,
    lib_dir:     Path,
    raw_dir:     Path,
) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_files = sorted(targets_dir.glob("*.json"))
    if not target_files:
        print(f"[ERROR] No target JSONs in {targets_dir} — run: "
              f"python src/parser.py --library <gates.sp>")
        sys.exit(1)

    targets = {}
    for tf in target_files:
        g = _build_full_graph(_load_json(tf))
        targets[tf.stem] = (g, target_radius(g, fallback_k))

    json_files = sorted(parsed_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] No entire-circuit JSONs in {parsed_dir}")
        sys.exit(1)

    rand_rate = float(negatives.get("random_per_pos",   1.5))
    part_rate = float(negatives.get("partial_per_pos",  0.5))
    mut_rate  = float(negatives.get("mutation_per_pos", 0.25))
    oth_rate  = float(negatives.get("others_per_pos",   0.25))

    tally = {k: 0 for k in ("pos", "rand", "part", "mut", "oth", "mut_ds_dropped")}
    file_idx = 0

    def _save(src, label, tag, tname):
        nonlocal file_idx
        d = Data(x=src.x, edge_index=src.edge_index, edge_type=src.edge_type)
        d.y = torch.tensor([label], dtype=torch.float)
        d.num_nodes = src.num_nodes
        d.target_name = tname
        torch.save(d, dataset_dir / f"{circuit}__{tname}__{file_idx:06d}_{tag}.pt")
        file_idx += 1

    vf3_out_dir = dataset_dir.parent / "vf3_out"
    vf3_out_dir.mkdir(parents=True, exist_ok=True)
    n_circuits = len(json_files)
    skipped = []

    for ci, json_path in enumerate(json_files, 1):
        circuit = json_path.stem
        jd = _load_json(json_path)
        full = _build_full_graph(jd)

        nt = full.node_types
        all_mosfets = ((nt == 3) | (nt == 4)).nonzero(as_tuple=True)[0].tolist()

        lib_sp = lib_dir / f"lib{circuit.lower()}.sp"
        raw_sp = raw_dir / f"{circuit}.sp"
        if not lib_sp.is_file() or not raw_sp.is_file():
            miss = lib_sp.name if not lib_sp.is_file() else raw_sp.name
            print(f"  [{ci}/{n_circuits}] {circuit}: SKIP (missing {miss})", flush=True)
            skipped.append(circuit)
            continue

        out_v = vf3_out_dir / f"{circuit}.v"
        print(f"  [{ci}/{n_circuits}] {circuit}: VF3 ({lib_sp.name})...", end="", flush=True)
        ok, msg = _run_vf3(vf3_bin, lib_sp, raw_sp, out_v)
        if not ok:
            print(f" FAILED — {msg}", flush=True)
            skipped.append(circuit)
            continue

        matches_by_target = parse_vf3_matches(out_v, jd)

        matches_by_target = {g: ms for g, ms in matches_by_target.items() if g in targets}

        if max_pos_per_target is not None:
            for g in matches_by_target:
                if len(matches_by_target[g]) > max_pos_per_target:
                    matches_by_target[g] = random.sample(
                        matches_by_target[g], max_pos_per_target)
        n_inst = sum(len(v) for v in matches_by_target.values())
        print(f" {n_inst} instances across {len(matches_by_target)} gate types", flush=True)

        for tname, (tgt_graph, K) in targets.items():
            matches = matches_by_target.get(tname, [])
            if not matches:
                continue

            matched_union = set().union(*matches) if matches else set()

            for mos_set in matches:
                center = next(iter(mos_set))
                pos_sub = _extract_subgraph(full, center, K)

                _save(pos_sub, 1.0, "pos", tname); tally["pos"] += 1

                non_match = [m for m in all_mosfets if m not in matched_union]
                n_rand = min(_expected_count(rand_rate), len(non_match))
                for nc in random.sample(non_match, n_rand) if n_rand else []:
                    rsub = _extract_subgraph(full, nc, K)
                    _save(rsub, 0.0, "neg_rand", tname); tally["rand"] += 1

                for _ in range(_expected_count(part_rate)):
                    pn = make_partial_negative(pos_sub)
                    if pn is not None:
                        _save(pn, 0.0, "neg_part", tname); tally["part"] += 1

                muts = make_mutation_negative(pos_sub)
                if muts:
                    n_mut = min(_expected_count(mut_rate), len(muts))
                    # Sample FIRST -- this preserves the original random stream exactly --
                    # then discard the drain/source variant.  Every other negative type is
                    # therefore bit-identical to the pre-fix dataset.
                    for variant, mv in random.sample(muts, n_mut):
                        if DROP_DS_MUTATION and variant == "ds":
                            tally["mut_ds_dropped"] += 1
                            continue
                        _save(mv, 0.0, f"neg_mut_{variant}", tname); tally["mut"] += 1

                for _ in range(_expected_count(oth_rate)):
                    other_names = [n for n in matches_by_target
                                   if n != tname and matches_by_target.get(n)]
                    if not other_names:
                        break
                    on = random.choice(other_names)
                    o_center = next(iter(random.choice(matches_by_target[on])))
                    o_sub = _extract_subgraph(full, o_center, K)
                    _save(o_sub, 0.0, "neg_oth", tname); tally["oth"] += 1

    total_neg = tally["rand"] + tally["part"] + tally["mut"] + tally["oth"]
    hard = tally["part"] + tally["mut"]
    sep = "─" * 52
    print(f"\n{sep}\n  Target-Conditioned Extraction Complete (VF3)\n{sep}")
    print(f"  Entire circuits    : {len(json_files)}" +
          (f"  ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""))
    print(f"  Target gates       : {len(targets)}  ({', '.join(targets)})")
    print(f"  Radii (K) per gate : " +
          ", ".join(f"{n}={targets[n][1]}" for n in targets))
    print(f"  Positives          : {tally['pos']}")
    print(f"  Neg random         : {tally['rand']}")
    print(f"  Neg partial [hard] : {tally['part']}")
    print(f"  Neg mutation[hard] : {tally['mut']}  (P/N type-swap only)")
    print(f"  Neg mut d/s DROPPED: {tally['mut_ds_dropped']}  "
          f"(mislabelled positives - see DROP_DS_MUTATION)")
    print(f"  Neg others         : {tally['oth']}")
    print(f"  Total negatives    : {total_neg}")
    if total_neg:
        print(f"  Hard fraction      : {hard / total_neg:.2%}")
    if tally["mut"]:
        print(f"  Partial:Mutation   : {tally['part'] / tally['mut']:.2f}:1")
    if tally["pos"]:
        print(f"  Effective neg:pos  : {total_neg / tally['pos']:.2f}:1")
    print(f"  Files written to   : {dataset_dir}\n{sep}\n")

if __name__ == "__main__":
    import yaml

    cfg_path = _PROJECT_DIR / "configs" / "config.yaml"
    if not cfg_path.is_file():
        print(f"[ERROR] config.yaml not found at {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ext_cfg = cfg.get("extractor", {})
    dat_cfg = cfg.get("data", {})

    negatives = ext_cfg.get("negatives") or {
        "random_per_pos": float(ext_cfg.get("neg_pos_ratio", 3)),
        "partial_per_pos": 0.0, "mutation_per_pos": 0.0, "others_per_pos": 0.0,
    }

    extract_dataset(
        parsed_dir         = _PROJECT_DIR / dat_cfg.get("parsed_dir",  "data/parsed"),
        targets_dir        = _PROJECT_DIR / dat_cfg.get("targets_dir", "data/parsed/targets"),
        dataset_dir        = _PROJECT_DIR / dat_cfg.get("dataset_dir", "data/dataset"),
        negatives          = negatives,
        max_pos_per_target = ext_cfg.get("max_pos_per_target", None),
        fallback_k         = ext_cfg.get("k_hops", 3),
        seed               = ext_cfg.get("seed",   42),
        vf3_bin            = _PROJECT_DIR / ext_cfg.get("vf3_bin", "vf3_cpp/build/prog"),
        lib_dir            = _PROJECT_DIR / ext_cfg.get("lib_dir", "vf3_cpp/examples/lib"),
        raw_dir            = _PROJECT_DIR / dat_cfg.get("raw_dir", "data/raw"),
    )