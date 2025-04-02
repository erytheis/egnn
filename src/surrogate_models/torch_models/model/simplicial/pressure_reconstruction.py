
from typing import List, Optional

import torch
import torch_geometric

from torch import scatter_add
from torch_geometric.graphgym import GCNConv
from torch_geometric.typing import OptTensor, Tensor, Adj

from src.surrogate_models.torch_models.model.metric import r2_score


class SimplicialNonParametricLayer(GCNConv):

    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, normalize=False)

    def forward(self, x: Tensor,
                edge_index: Tensor, weights: OptTensor,
                size=None,
                *args, **kwargs
                ):
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=weights,
                             size=size, **kwargs)

        return out

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim) and self.message_passing_direction == 'boundary':
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))


class SimplicialFeaturePropagation(GCNConv):
    def __init__(self, *args, **kwargs, ):
        super().__init__(in_channels=1, out_channels=1, *args, **kwargs, normalize=False)
        self.lin = None

    
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class PowerApproximationLayer(torch.nn.Module):

    def __init__(self, num_iterations=50, virtual_nodes=True, aggr=None, **kwargs):
        self.num_iterations = num_iterations
        self.virtual_nodes = virtual_nodes

        self.matmul = SimplicialNonParametricLayer(1, 1, flow='target_to_source').eval()
        self.fp = SimplicialFeaturePropagation(node_dim=0, flow='target_to_source').eval()

    
    def scatter_laplacian(self, batch, output_, node_sensor_idx=3, junction_idx=2, epsilon=0.0001):
        total_evaluation_time = 0

        output_['x'] = torch.zeros_like(batch.node_y.unsqueeze(-1))
        output = output_['edge_attr'].squeeze()

        # add known heads
        known_nodes = batch.x[:, node_sensor_idx] == 1
        unknown_nodes = batch.x[:, junction_idx] == 1
        known_heads = batch.node_y[known_nodes].unsqueeze(-1)
        real_nodes = batch.x[:, -1] == 0
        real_edges = batch.edge_attr[:, -1] == 0

        # scatter operation of B1 @ known_heads
        dH_known = self.matmul(batch.x[:, 1].unsqueeze(-1), batch.boundary_index.flip(0),
                               batch.boundary_weight,
                               size=tuple([i + 1 for i in batch.boundary_index.flip(0).max(dim=1)[0]]))
        # dH_known = (B1_known @ known_heads)

        loss_coefficient = batch.edge_attr[:, 1]
        dH_predicted = torch.abs(output) ** 1.852 * loss_coefficient * torch.sign(output)

        # get right side
        right_side = self.matmul(dH_predicted.unsqueeze(-1) - dH_known, batch.boundary_index,
                                 batch.boundary_weight, size=tuple([i + 1 for i in batch.boundary_index.max(dim=1)[0]]))
        right_side[known_nodes] = 0
        right_side = right_side  # [unknown_nodes]
        right_side = right_side.squeeze()

        num_nodes = batch.num_nodes
        edge_index = torch.cat([batch.edge_index[:, real_edges], batch.edge_index.flip(0)[:, real_edges]], axis=1)
        row, col = edge_index[0], edge_index[1]
        edge_weight = torch.ones(edge_index.size(1),  # dtype=dtype,
                                 device=edge_index.device)

        # find laplacian
        laplacian_torch_ = torch_geometric.utils.get_laplacian(edge_index, normalization='rw')

        # find degrees
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # [unknown_nodes]
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        # find solution dense
        dense_solution = torch.zeros(batch.num_nodes, device=batch.x.device)
        dense_solution[known_nodes] = known_heads.squeeze()
        dense_solution = dense_solution[real_nodes]

        # mask edges adjacent to sensor nodes
        sensors = batch.x[:, node_sensor_idx] == 1
        sensor_edges = sensors[laplacian_torch_[0][0]] + sensors[laplacian_torch_[0][1]]
        unknown_edges = real_nodes[laplacian_torch_[0][0]] + real_nodes[laplacian_torch_[0][1]]
        laplacian_torch_[1][sensor_edges] = 0
        laplacian_torch_[1][sensor_edges] = 0
        scatter_solution = torch.zeros(batch.num_nodes, device=batch.x.device)

        right_side = (right_side * deg_inv)
        l_scatter_weights = right_side.clone().unsqueeze(-1)
        l_scatter_output = torch.zeros_like(right_side).unsqueeze(-1)
        # l_iter = torch.zeros_like(laplacian_torch)

        # add identity matrix
        N = batch.num_nodes
        loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        # remove eye from laplacian
        mask = (laplacian_torch_[0][0] == laplacian_torch_[0][1])
        laplacian_torch_[1][mask] = laplacian_torch_[1][mask] - 1
        indices = laplacian_torch_[0][:, unknown_edges]
        weights = -laplacian_torch_[1][unknown_edges]

        # truncated solution
        for i in range(50000):
            # multiply with identity matrix
            if i == 0:
                l_scatter_weights = self.fp(
                    l_scatter_weights,
                    loop_index[:, unknown_nodes],
                    torch.ones(N, device=edge_index.device)[unknown_nodes]
                )
            else:
                l_scatter_weights = self.fp(
                    l_scatter_weights,
                    indices,
                    weights,
                )

            # convergence condition
            time_st = time.time()
            if i % 1000 == 0:
                norm = torch.norm(l_scatter_weights, dim=0)
                print('norm:', norm)
                if norm < epsilon:
                    print('converged')
                    break

                virtual = batch.x[:, -1] == 1
                acc = r2_score(batch.node_y[~virtual], l_scatter_output.squeeze()[~virtual])
                print(acc)
                if acc > 0.99:
                    print('converged')
                    break
            time_end = time.time()
            evaluation_time = time_end - time_st
            total_evaluation_time += evaluation_time

            l_scatter_output += l_scatter_weights
        scatter_solution = l_scatter_output.squeeze()

        output_['x'] = scatter_solution.unsqueeze(-1)
        return output_, total_evaluation_time



def get_heads_from_flowrates_linalg(batch, output, dH_predicted=None,
                                    reservoir_idx=3,
                                    junction_idx=2,
                                    edge_mask=None):
    if dH_predicted is None:
        loss_coefficient = batch.edge_attr[:, 1]
        dH_predicted = torch.abs(output) ** 1.852 * loss_coefficient * torch.sign(output)

    B1 = torch.sparse_coo_tensor(batch.boundary_index, batch.boundary_weight,
                                 size=(batch.num_nodes, batch.num_edges))
    B1_ = B1.to_dense()

    # remove virtual_edges
    B1_ = B1_[:, edge_mask]

    # add known heads
    known_nodes = batch.x[:, reservoir_idx] == 1
    unknown_nodes = batch.x[:, junction_idx] == 1
    B1_known = B1_[known_nodes].T
    known_heads = batch.node_y[known_nodes].unsqueeze(-1)
    # known_heads = batch.x[known_nodes, 1].unsqueeze(-1)

    dH_known = (B1_known @ known_heads).squeeze()

    # get prediction
    solution = torch.zeros(batch.num_nodes, device=batch.x.device)
    solution[known_nodes] = known_heads.squeeze()
    #
    solution[unknown_nodes] = torch.linalg.lstsq(B1_[unknown_nodes].T, dH_predicted - dH_known,
                                                 driver='gels',
                                                 ).solution.squeeze()

    return solution



def from_flowrates_to_heads(batch, output_, node_sensor_idx=3, **kwargs):
    output_['x'] = torch.zeros_like(batch.node_y.unsqueeze(-1))

    for i in range(len(batch)):
        batch_ = batch[i]

        # get positioning
        edge_mask = batch_.mask_by_features(value=0, column_idx=2)
        # node_mask = batch_.mask_by_features(value=0, column_idx=4, attribute='x')

        slices_x = slice(batch.slices['x'][i], batch.slices['x'][i + 1])
        slices_edge_attr = slice(batch.slices['edge_attr'][i], batch.slices['edge_attr'][i + 1])

        # remove virtual edges
        output = output_['edge_attr'][slices_edge_attr]
        output = output[edge_mask].squeeze()
        # batch_.drop_node_by_features(nodes_value=0, edges_value=0, edges_column_idx=2)

        # get edge prediction
        loss_coefficient = batch_.edge_attr[:, 1][edge_mask]
        dH_predicted = torch.abs(output) ** 1.852 * loss_coefficient * torch.sign(output)

        # get true values
        heads = batch_.x[:, 1]

        # derive node-prediction
        node_prediction = get_heads_from_flowrates_linalg(batch_, output, dH_predicted, node_sensor_idx,
                                                          edge_mask=edge_mask)

        output_['x'][slices_x] = node_prediction.unsqueeze(-1)

    return output_
