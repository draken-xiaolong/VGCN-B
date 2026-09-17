"""Exercise early/late loss schedules on synthetic graphs, without map data.

This is a training-code smoke test, not a scientific reproduction experiment.
It does not change the supplied checkpoint.
"""
import json
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'VGCN'))
from VGCN import GCNModel, ContrastiveTrainer, set_global_seed


def main():
    set_global_seed(42)
    torch.set_num_threads(2)
    model = GCNModel(input_dim=13)
    trainer = ContrastiveTrainer(model, device='cpu', batch_size=4)
    originals, attacked = {}, {}
    for c in range(3):
        n = 16 + 4*c
        nodes = torch.arange(n)
        successors = (nodes+1) % n
        edges = torch.stack([torch.cat([nodes,successors]),torch.cat([successors,nodes])])
        graph = Data(x=torch.randn(n,13)+c,edge_index=edges)
        originals[str(c)] = graph
        attacked[str(c)] = [Data(x=graph.x+0.02*torch.randn_like(graph.x),edge_index=edges) for _ in range(2)]
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    reports = []
    for epoch in (0,11):
        metrics = trainer.train_epoch(originals,attacked,epoch)
        assert torch.isfinite(torch.tensor(metrics)).all(), 'Non-finite training metric'
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        reports.append({'schedule_epoch':epoch,'total_loss':float(metrics[0])})
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items()), 'Model did not update'
    print(json.dumps({'synthetic_training_smoke':'passed','full_retraining':False,'schedule_checks':reports},indent=2))


if __name__ == '__main__':
    main()
