import os
import re
import sys
import random
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent

sys.path.insert(0, str(_SRC_DIR))
from model import CircuitFilterGNN, CHECKPOINTS_DIR

def load_config(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        print(f"[ERROR] config.yaml not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def _clean_graph(g, keep_label: bool):

    d = Data(x=g.x, edge_index=g.edge_index, edge_type=g.edge_type)
    d.num_nodes = g.num_nodes
    if keep_label:
        d.y = g.y.view(1).float()
    return d

def load_targets(targets_dir: Path) -> dict:

    sys.path.insert(0, str(_SRC_DIR))
    from extractor import _build_full_graph, _load_json

    files = sorted(targets_dir.glob("*.json"))
    if not files:
        print(f"[ERROR] No target graphs in {targets_dir}")
        print(f"        Run: python src/parser.py --library <gates.sp> --targets-dir {targets_dir}")
        sys.exit(1)

    targets = {}
    for f in files:
        targets[f.stem] = _clean_graph(_build_full_graph(_load_json(f)), keep_label=False)
    return targets

def load_all_pt_files(dataset_dir: Path) -> list:

    pt_files = sorted(dataset_dir.glob("*.pt"))
    if not pt_files:
        print(f"[ERROR] No .pt files found in {dataset_dir}")
        print(f"        Run extractor.py first: python src/extractor.py")
        sys.exit(1)

    samples = []
    missing_tag = 0
    for p in pt_files:
        g = torch.load(p, weights_only=False)
        if not hasattr(g, "y"):
            continue
        tname = getattr(g, "target_name", None)
        if tname is None:
            missing_tag += 1
            continue
        samples.append((_clean_graph(g, keep_label=True), tname, p.name))

    if missing_tag:
        print(f"[WARN] {missing_tag} .pt files had no .target_name and were skipped.")
        print(f"       Those are old single-graph samples; regenerate with the "
              f"target-conditioned extractor.")
    if not samples:
        print(f"[ERROR] No target-tagged samples found in {dataset_dir}.")
        print(f"        Regenerate the dataset: python src/extractor.py")
        sys.exit(1)
    return samples

class PairDataset(Dataset):

    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        cand, tname = self.samples[i][0], self.samples[i][1]
        return cand, self.targets[tname]

def collate_pairs(batch):

    cands, tgts = zip(*batch)
    return Batch.from_data_list(list(cands)), Batch.from_data_list(list(tgts))

_CKT_RE = re.compile(r'^(.*?)__')

def split_dataset(
    graphs: list,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """Leave-circuits-out split.

    The previous implementation shuffled the flat sample list.  Because extractor.py
    derives `neg_part` (parent minus ONE transistor) and `neg_mut` (identical node set
    and edge_index, only edge_type perturbed) FROM a specific positive, a flat shuffle
    scatters a positive and its own perturbations across different splits.

    Measured on this dataset (38,939 samples, seed 42, 70/15/15):
        derived negatives (neg_part + neg_mut) : 8,304
        separated from their parent positive   : 3,946  (47.5%)
    i.e. 33.8% of val+test were near-duplicates of graphs seen during training, and all
    25 circuits appeared in all three splits.  Reported recall was inflated accordingly.

    Splitting by circuit also removes the overlap between K-hop regions centred on
    adjacent transistors of the same netlist.

    NOTE: with only 25 circuits this split is coarse and higher-variance than the old
    one.  Run several seeds and report mean +/- std, not a single number.
    """
    by_ckt: dict = {}
    for item in graphs:
        fname = item[2] if len(item) > 2 else ""
        m = _CKT_RE.match(fname)
        by_ckt.setdefault(m.group(1) if m else "UNKNOWN", []).append(item)

    ckts = sorted(by_ckt)
    if len(ckts) < 3:
        print(f"[WARN] only {len(ckts)} circuit group(s) found ({', '.join(ckts)}) -"
              f" falling back to a flat shuffle. Expected filenames of the form"
              f" '<CIRCUIT>__<TARGET>__NNNNNN_tag.pt'.")
        shuffled = list(graphs)
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_tr, n_va = int(n * train_frac), int(n * val_frac)
        return shuffled[:n_tr], shuffled[n_tr:n_tr + n_va], shuffled[n_tr + n_va:]

    random.Random(seed).shuffle(ckts)
    n = len(ckts)
    n_train = min(max(1, round(n * train_frac)), n - 2)
    n_val   = min(max(1, round(n * val_frac)),   n - n_train - 1)

    tr_c, va_c, te_c = ckts[:n_train], ckts[n_train:n_train + n_val], ckts[n_train + n_val:]
    train_set = [x for c in tr_c for x in by_ckt[c]]
    val_set   = [x for c in va_c for x in by_ckt[c]]
    test_set  = [x for c in te_c for x in by_ckt[c]]

    print(f"  Split             : leave-circuits-out (seed {seed})")
    print(f"    train {len(tr_c):>2} ckts ({len(train_set):>6} samples): {', '.join(tr_c)}")
    print(f"    val   {len(va_c):>2} ckts ({len(val_set):>6} samples): {', '.join(va_c)}")
    print(f"    test  {len(te_c):>2} ckts ({len(test_set):>6} samples): {', '.join(te_c)}")
    return train_set, val_set, test_set

def resolve_pos_weight(
    cfg_value,
    train_graphs: list[Data],
    device: torch.device,
) -> torch.Tensor:

    if cfg_value is not None:
        w = float(cfg_value)
        print(f"  pos_weight        : {w:.2f}  (from config)")
        return torch.tensor([w], device=device)

    num_pos = sum(1 for cand, *_ in train_graphs if cand.y.item() == 1)
    num_neg = sum(1 for cand, *_ in train_graphs if cand.y.item() == 0)

    if num_pos == 0:
        print("[WARN] No positive examples in training split — defaulting pos_weight to 1.0")
        return torch.tensor([1.0], device=device)

    w = float(num_neg / num_pos)
    print(f"  pos_weight        : {w:.2f}  (dynamic: {num_neg} neg / {num_pos} pos)")
    return torch.tensor([w], device=device)

def build_scheduler(optimizer, sch_cfg: dict):

    if not sch_cfg or not sch_cfg.get("enabled", False):
        return None

    sch_type = sch_cfg.get("type", "plateau")
    if sch_type != "plateau":
        print(f"[WARN] Unknown scheduler type '{sch_type}' — falling back to plateau")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "max",
        factor    = float(sch_cfg.get("factor",   0.5)),
        patience  = int(sch_cfg.get("patience",   15)),
        min_lr    = float(sch_cfg.get("min_lr",   1e-5)),
        threshold = float(sch_cfg.get("threshold", 1e-3)),
    )

def warmup_lr_factor(epoch: int, warmup_epochs: int) -> float:

    if warmup_epochs and warmup_epochs > 0 and epoch <= warmup_epochs:
        return epoch / float(warmup_epochs)
    return 1.0

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:

    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    for cand, tgt in loader:
        cand = cand.to(device)
        tgt  = tgt.to(device)
        logits = model(cand, tgt)
        labels = cand.y.view(-1, 1)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * cand.num_graphs

        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()
        labels = cand.y.long()

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    n     = tp + fp + fn + tn
    avg_loss = total_loss / max(n, 1)
    acc      = (tp + tn) / max(n, 1)

    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-8)

    model.train()
    return avg_loss, acc, prec, rec, f1

def train(cfg_path: Path) -> None:
    cfg      = load_config(cfg_path)
    dat_cfg  = cfg.get("data",     {})
    mdl_cfg  = cfg.get("model",    {})
    trn_cfg  = cfg.get("training", {})
    ext_cfg  = cfg.get("extractor", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = _PROJECT_DIR / dat_cfg.get("dataset_dir", "data/dataset")
    targets_dir = _PROJECT_DIR / dat_cfg.get("targets_dir", "data/parsed/targets")

    targets = load_targets(targets_dir)
    samples = load_all_pt_files(dataset_dir)

    train_graphs, val_graphs, test_graphs = split_dataset(
        samples,
        train_frac = dat_cfg.get("train_split", 0.70),
        val_frac   = dat_cfg.get("val_split",   0.15),
        seed       = ext_cfg.get("seed",        42),
    )

    batch_size = trn_cfg.get("batch_size", 64)
    train_loader = DataLoader(PairDataset(train_graphs, targets), batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_pairs)
    val_loader   = DataLoader(PairDataset(val_graphs, targets),   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pairs)
    test_loader  = DataLoader(PairDataset(test_graphs, targets),  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_pairs)

    model = CircuitFilterGNN(
        in_channels     = mdl_cfg.get("in_channels",     5),
        hidden_channels = mdl_cfg.get("hidden_channels", 128),
        num_relations   = mdl_cfg.get("num_relations",   22),
        num_bases       = mdl_cfg.get("num_bases",        8),
        num_layers      = mdl_cfg.get("num_layers",       2),
    ).to(device)

    pos_weight = resolve_pos_weight(
        cfg_value    = trn_cfg.get("pos_weight", None),
        train_graphs = train_graphs,
        device       = device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    base_lr   = trn_cfg.get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = base_lr,
        weight_decay = trn_cfg.get("weight_decay",  0.0001),
    )

    scheduler     = build_scheduler(optimizer, trn_cfg.get("scheduler", {}))
    warmup_epochs = int(trn_cfg.get("warmup_epochs", 0) or 0)

    grad_clip_norm = trn_cfg.get("grad_clip_norm", None)
    if grad_clip_norm in (0, 0.0):
        grad_clip_norm = None
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    es_cfg          = trn_cfg.get("early_stopping", {}) or {}
    es_enabled      = bool(es_cfg.get("enabled", False))
    es_patience     = int(es_cfg.get("patience", 60))
    es_min_delta    = float(es_cfg.get("min_delta", 0.0005))
    epochs_no_improve = 0

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt  = CHECKPOINTS_DIR / "best_model.pt"
    best_val_f1 = 0.0

    checkpoint_metric = str(trn_cfg.get("checkpoint_metric", "f1")).lower()
    if checkpoint_metric not in ("f1", "recall"):
        print(f"[WARN] Unknown checkpoint_metric '{checkpoint_metric}' — using 'f1'")
        checkpoint_metric = "f1"
    best_val_score = 0.0

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_train_pos = sum(1 for cand, *_ in train_graphs if cand.y.item() == 1)
    n_train_neg = sum(1 for cand, *_ in train_graphs if cand.y.item() == 0)

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  CircuitFilterGNN  —  Training")
    print(sep)
    print(f"  Device            : {device}")
    print(f"  Total graphs      : {len(samples)}")
    print(f"  Target gates      : {len(targets)}  ({', '.join(sorted(targets))})")
    print(f"  Train / Val / Test: {len(train_graphs)} / {len(val_graphs)} / {len(test_graphs)}")
    print(f"  Train pos/neg     : {n_train_pos} / {n_train_neg}")
    print(f"  Model parameters  : {total_params:,}")
    print(f"  Epochs            : {trn_cfg.get('epochs', 50)}")
    print(f"  Batch size        : {batch_size}")
    print(f"  Learning rate     : {trn_cfg.get('learning_rate', 0.001)}")
    print(f"  Warmup epochs     : {warmup_epochs}")
    print(f"  Grad clip norm    : {grad_clip_norm if grad_clip_norm is not None else 'off'}")
    print(f"  LR scheduler      : {'ReduceLROnPlateau (val_f1)' if scheduler is not None else 'off'}")
    print(f"  Early stopping    : {'on (patience=' + str(es_patience) + ')' if es_enabled else 'off'}")
    print(f"  Checkpoint metric : val_{checkpoint_metric}")
    print(f"  Checkpoint target : {best_ckpt}")
    print(f"{sep}\n")

    epochs = trn_cfg.get("epochs", 50)
    model.train()

    for epoch in range(1, epochs + 1):

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            wf = warmup_lr_factor(epoch, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * wf

        epoch_loss = 0.0
        num_graphs = 0

        for cand, tgt in train_loader:
            cand = cand.to(device)
            tgt  = tgt.to(device)
            optimizer.zero_grad()
            logits = model(cand, tgt)
            loss   = criterion(logits, cand.y.view(-1, 1))
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * cand.num_graphs
            num_graphs += cand.num_graphs

        avg_train_loss = epoch_loss / max(num_graphs, 1)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)
        val_score = val_rec if checkpoint_metric == "recall" else val_f1

        improved = val_score > (best_val_score + es_min_delta)
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_f1 = val_f1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      val_f1,
                "val_recall":  val_rec,
                "val_prec":    val_prec,
                "val_acc":     val_acc,
                "metric":      checkpoint_metric,
                "cfg":         cfg,
            }, best_ckpt)
            ckpt_marker = "  ✓ saved"
        else:
            ckpt_marker = ""

        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step(val_score)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>3}/{epochs}"
            f"  lr={current_lr:.2e}"
            f"  train_loss={avg_train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  P={val_prec:.3f}"
            f"  R={val_rec:.3f}"
            f"  F1={val_f1:.3f}"
            f"{ckpt_marker}"
        )

        if es_enabled and epoch > warmup_epochs:
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= es_patience:
                    print(
                        f"\n  Early stopping at epoch {epoch} "
                        f"— no val_{checkpoint_metric} gain > {es_min_delta} for "
                        f"{es_patience} epochs (best val_{checkpoint_metric}={best_val_score:.4f})."
                    )
                    break

    print(f"\n{sep}")
    print(f"  Final Test Evaluation (best checkpoint)")
    print(sep)

    if not best_ckpt.is_file():
        print(f"  [WARN] No checkpoint was saved (val_{checkpoint_metric} never improved above 0).")
        print(f"         Skipping test evaluation.")
        print(f"{sep}\n")
        return

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"  Checkpoint epoch  : {ckpt['epoch']}")
    print(f"  Best val {checkpoint_metric:<8}: {ckpt.get('val_' + checkpoint_metric, ckpt['val_f1']):.4f}")
    print(f"  Test loss         : {test_loss:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Test precision    : {test_prec:.4f}")
    print(f"  Test recall       : {test_rec:.4f}   (← gates kept; 1.0 = none missed)")
    print(f"  Test F1           : {test_f1:.4f}")
    print(f"{sep}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CircuitFilterGNN")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_PROJECT_DIR / "configs" / "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    train(Path(args.config))