import torch
from line_profiler_pycharm import profile
from torch_geometric.data import Batch

from src.surrogate_models.torch_models.model.base_gnn import WeightAggregation


class PresssureReconstruction():
    differentiable: bool

    def __init__(self,
                 node_sensor_idx=3,
                 virtual_nodes_idx=-1,
                 virtual_edges_idx=-1,
                 **kwargs):
        self.node_sensor_idx = node_sensor_idx
        self.virtual_nodes_idx = virtual_nodes_idx
        self.virtual_edges_idx = virtual_edges_idx

        pass

    def forward(self, batch, output):
        raise NotImplementedError


class ShortestPathDistance(PresssureReconstruction):
    differentiable: bool = True

    def __init__(self, num_iterations: int = 50,

                 **kwargs):
        self.num_iterations = num_iterations
        super().__init__(**kwargs)

    def forward(self, batch, output, **kwargs):
        # override the number of iterations if it is passed as a keyword argument
        num_iterations = self.num_iterations if 'num_iterations' not in kwargs else kwargs['num_iterations']

        virtual_nodes = True if self.virtual_nodes_idx is not None else None

        return iterative_shortest_path_distance(batch, output,
                                                num_iterations=num_iterations,
                                                virtual_nodes=virtual_nodes,
                                                **kwargs)


def distance_from_sources(edge_index,
                          source_node_values,  # vector with values of the sources nodes and zeros for the rest
                          destination_nodes,  # vector with the indices of the destination nodes
                          aggr=None,
                          weights=None):
    # get the sensor values
    if aggr is None:
        aggr = WeightAggregation(aggr='mean')

    p_ = source_node_values.clone()
    W = weights if weights is not None else torch.ones(edge_index.shape[1], dtype=torch.float32)
    num_nodes = source_node_values.shape[0]

    source_nodes = (source_node_values > 0).float()  # vector with ones for the source nodes and zeros for the rest
    check = torch.zeros_like(source_nodes).bool()

    while not torch.any(check) > 0:
        # edges incident to all the nodes that have a source node as a source
        source_edges_src = source_nodes[edge_index[0]]
        source_edges_dst = source_nodes[edge_index[1]]

        # get edges that are incident to one source node
        source_edges = source_edges_src + source_edges_dst == 1

        # get the edges that need to be flipped
        to_flip = torch.stack([source_edges_dst.bool(), ~source_edges_src.bool()])
        to_flip = torch.all(to_flip, dim=0)

        # flip the edges
        edge_index[:, to_flip] = edge_index[:, to_flip].flip(0)
        W[to_flip] = W[to_flip] * -1

        # contaminate new nodes
        p_new = aggr(edge_index[:, source_edges], p_.unsqueeze(-1),
                     edge_weight=W[source_edges].abs().unsqueeze(-1),
                     size=(num_nodes, num_nodes)
                     ).squeeze()

        p_ += p_new  # + p_visited

        # update the source nodes
        source_nodes_ = (source_nodes.clone() > 0).float()
        new_source_nodes = torch.zeros_like(source_nodes)

        source_nodes += (
                torch.scatter_add(new_source_nodes, 0, edge_index[1], source_nodes_[edge_index[0]]) > 0).float()
        source_nodes += (
                torch.scatter_add(new_source_nodes, 0, edge_index[0], source_nodes_[edge_index[1]]) > 0).float()

        check = torch.all(torch.stack([destination_nodes, source_nodes]), axis=0)

    return p_


def nearest_neighbour(edge_index,
                      source_nodes,  # vector with values of the sources nodes and zeros for the rest
                      destination_nodes,  # vector with the indices of the destination nodes
                      condition_function='all',
                      ):
    edge_index = edge_index.clone()
    source_nodes = source_nodes.float()
    condition_function = torch.any if condition_function == 'any' else torch.all
    check = torch.zeros_like(source_nodes).bool()

    output = torch.zeros_like(source_nodes)

    while True:
        old_source_nodes = source_nodes.clone()

        # remove source nodes that are already destination nodes and store them
        output[torch.all(torch.stack([destination_nodes, source_nodes]), axis=0)] = 1
        source_nodes[destination_nodes] = 0

        # edges incident to all the nodes that have a source node as a source
        source_edges_src = source_nodes[edge_index[0]]
        source_edges_dst = source_nodes[edge_index[1]]

        # get the edges that need to be flipped
        to_flip = torch.stack([source_edges_dst.bool(), ~source_edges_src.bool()])
        to_flip = torch.all(to_flip, dim=0)

        # flip the edges
        edge_index[:, to_flip] = edge_index[:, to_flip].flip(0)

        # update the source nodes
        source_nodes_ = (source_nodes.clone() > 0).float()
        _ = torch.zeros_like(source_nodes)

        # new source nodes
        source_nodes += (
                torch.scatter_add(_, 0, edge_index[1], source_nodes_[edge_index[0]]) > 0).float()
        source_nodes += (
                torch.scatter_add(_, 0, edge_index[0], source_nodes_[edge_index[1]]) > 0).float()

        source_nodes = source_nodes.bool().float()

        if torch.all(old_source_nodes == source_nodes):
            return output.bool()


@profile
def iterative_shortest_path_distance(batch, output_, node_sensor_idx=3, num_iterations=50, virtual_nodes=True,
                                     aggr=None, **kwargs):
    if aggr is None:
        aggr = WeightAggregation(aggr='mean')

    edge_index = batch.edge_index
    edge_attr = batch.edge_attr

    source_nodes = batch.x[:, node_sensor_idx].clone()

    if virtual_nodes:
        virtual_nodes = batch.x[:, -1] == 1
        virtual_edges = batch.edge_attr[:, -1] == 1

    else:
        virtual_nodes = torch.zeros(batch.num_nodes, dtype=torch.bool)
        virtual_edges = torch.zeros(batch.num_edges, dtype=torch.bool)

    edge_index = batch.edge_index[:, ~virtual_edges].clone()

    # get the edge weights
    W = batch.edge_attr[:, 1] * output_['edge_attr'].abs().squeeze().clone() ** 1.852
    # W = batch.edge_attr[:, 1] * batch.edge_y.abs().squeeze().clone() ** 1.852
    W = W[~virtual_edges]
    W = W.squeeze() if W.dim() == 2 else W

    # get the sensor values
    initial_node_value = batch.x[:, 1]
    target = batch.node_y  # [~virtual_nodes]
    p_ = initial_node_value.clone()  # [~virtual_nodes]

    for i in range(num_iterations):
        # old souce nodes and node values

        # edges incident to all the nodes that have a source node as a source
        source_edges_src = source_nodes[edge_index[0]]
        source_edges_dst = source_nodes[edge_index[1]]

        # get edges that are incident to one source node
        source_edges = source_edges_src + source_edges_dst == 1
        # get edges that are incident to two source nodes
        visited_edges = source_edges_src + source_edges_dst > 1

        # get the edges that need to be flipped
        to_flip = torch.stack([source_edges_dst.bool(), ~source_edges_src.bool()])
        to_flip = torch.all(to_flip, dim=0)

        # flip the edges
        edge_index[:, to_flip] = edge_index[:, to_flip].flip(0)
        # W[to_flip] = W[to_flip] * -1

        # contaminate new nodes
        num_nodes = batch.num_nodes  # - virtual_nodes.sum()
        p_new = aggr(edge_index[:, source_edges], p_.unsqueeze(-1),
                     edge_weight=W[source_edges].abs().unsqueeze(-1),
                     size=(num_nodes, num_nodes)
                     ).squeeze()

        p_ += p_new  # + p_visited

        # update the source nodes
        source_nodes_ = (source_nodes.clone() > 0).float()
        new_source_nodes = torch.zeros_like(source_nodes)

        source_nodes += (
                torch.scatter_add(new_source_nodes, 0, edge_index[1], source_nodes_[edge_index[0]]) > 0).float()
        source_nodes += (
                torch.scatter_add(new_source_nodes, 0, edge_index[0], source_nodes_[edge_index[1]]) > 0).float()

    output_['x'] = p_.unsqueeze(-1)
    return output_


@profile
def get_heads_from_flowrates_linalg(batch, output, dH_predicted=None,
                                    reservoir_idx=3,
                                    junction_idx=2,
                                    edge_mask=None):
    # real_nodes = batch.x[:, -1] == 1
    # real_edges = batch.edge_attr[:, -1] == 1

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


@profile
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
        dH_true = heads[batch_.edge_index[1]] - heads[batch_.edge_index[0]]

        # derive node-prediction
        node_prediction = get_heads_from_flowrates_linalg(batch_, output, dH_predicted, node_sensor_idx,
                                                          edge_mask=edge_mask)

        output_['x'][slices_x] = node_prediction.unsqueeze(-1)

    return output_

# reservoir_nodes = batch_.mask_by_features(value=1, column_idx=3, attribute='x')
# reservoir_pipes = (reservoir_nodes[batch_.edge_index[0]] | reservoir_nodes[batch_.edge_index[1]])[
#     edge_mask]
# output[reservoir_pipes] = batch_.edge_y[edge_mask][reservoir_pipes]