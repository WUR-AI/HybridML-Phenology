# import os
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import pandas as pd
#
# import config
# from data.bloom_doy import get_locations
#
#
# # TODO -- save parameters to df
#
#
# class ParameterModel(nn.Module):
#
#     def get_parameters(self, xs: dict) -> tuple:
#         raise NotImplementedError
#
#
# class AccumulationParameterMapping(ParameterModel):
#
#     def __init__(self,
#                  location_groups: dict,
#                  init_val_th_c: float = 0.0,
#                  init_val_th_g: float = 0.0,
#                  init_val_tb_g: float = 0.0,
#                  ):
#         super().__init__()
#
#         self._thc_map = ParameterMapping(
#             location_groups,
#             init_val=init_val_th_c,
#         )
#
#         self._thg_map = ParameterMapping(
#             location_groups,
#             init_val=init_val_th_g,
#         )
#
#         self._tbg_map = ParameterMapping(
#             location_groups,
#             init_val=init_val_tb_g,
#         )
#
#     def get_parameters(self, xs: dict) -> tuple:
#         return self(xs)
#
#     def forward(self, xs: dict):
#
#         thc = self._thc_map(xs)
#         thg = self._thg_map(xs)
#         tbg = self._tbg_map(xs)
#
#         thc = self._scale_param(thc, 1)
#         thg = self._scale_param(thg, 1)
#         # tbg = self._scale_param(tbg, 20)
#         tbg = self._scale_param(tbg, 1)
#
#         return thc, thg, tbg
#
#     def ungroup(self) -> None:
#         """
#         Replace the current parameter mappings by ones without location grouping (in-place)
#         :return: None
#         """
#         self._thc_map = self._thc_map.as_ungrouped()
#         self._thg_map = self._thg_map.as_ungrouped()
#         self._tbg_map = self._tbg_map.as_ungrouped()
#
#     @staticmethod
#     def _scale_param(p: torch.Tensor, c) -> torch.Tensor:
#         p = _modified_abs(p)
#         return p * c
#
#
# class LocalAccumulationParameterMapping(AccumulationParameterMapping):
#
#     def __init__(self,
#                  locations: list,
#                  init_val_th_c: float = 0.0,
#                  init_val_th_g: float = 0.0,
#                  init_val_tb_g: float = 0.0,
#                  ):
#         super().__init__(
#             {loc: i for i, loc in enumerate(locations)},
#             init_val_th_c,
#             init_val_th_g,
#             init_val_tb_g,
#         )
#
#
# class GlobalAccumulationParameterMapping(AccumulationParameterMapping):
#
#     def __init__(self,
#                  locations: list,
#                  init_val_th_c: float = 0.0,
#                  init_val_th_g: float = 0.0,
#                  init_val_tb_g: float = 0.0,
#                  ):
#         super().__init__(
#             {loc: 0 for loc in locations},
#             init_val_th_c,
#             init_val_th_g,
#             init_val_tb_g,
#         )
#
#
# class ParameterMapping(nn.Module):
#
#     def __init__(self,
#                  location_groups: dict,
#                  init_val: float = 0.0,
#                  init_val_group: dict = None,
#                  ):
#         super().__init__()
#         # Make sure all group are assigned an initial value
#         assert init_val_group is None or all([group in init_val_group.keys() for group in set(location_groups.values())])
#
#         n_groups = len(set(location_groups.values()))
#
#         # Rename all groups to some index
#         group_to_ix = {
#             g: i for i, g in enumerate(set(location_groups.values()))
#         }
#
#         # Map all locations to these indices
#         self._loc_ixs = {
#             loc: group_to_ix[g] for loc, g in location_groups.items()
#         }
#
#         if init_val_group is None:
#             self._ps = nn.ParameterList(
#                 [torch.tensor(float(init_val), requires_grad=True) for _ in range(n_groups)]
#             )
#         else:
#             self._ps = nn.ParameterList(
#                 [torch.tensor(float(init_val_group[g]), requires_grad=True) for g in group_to_ix.keys()]
#             )
#
#     @property
#     def locations(self):
#         return list(self._loc_ixs.keys())
#
#     @property
#     def location_group_ixs(self) -> dict:
#         return dict(self._loc_ixs)
#
#     @property
#     def location_values(self) -> dict:
#         return {
#             loc: self._ps[ix].item() for loc, ix in self.location_group_ixs.items()
#         }
#
#     def forward(self, xs: dict):
#         locations = xs['location']
#
#         ps = torch.cat([self._ps[self._loc_ixs[loc]].view(1, 1) for loc in locations],
#                        dim=0,
#                        )
#
#         return ps
#
#     def as_ungrouped(self) -> 'ParameterMapping':
#         """
#         Recreate the current parameter mapping without grouped locations
#         That is, each location is assigned one value (the one of the group its currently assigned to)
#         :return: a new ParameterMapping object without location groups
#         """
#         # Assign an index (group) to all locations individually
#         loc_ixs = {loc: ix for ix, loc in enumerate(self.locations)}
#         # Get the values that are currently assigned to all locations
#         loc_vals = self.location_values
#         # Create a new parameter mapping without location grouping
#         return ParameterMapping(
#             location_groups=loc_ixs,
#             init_val_group={ix: loc_vals[loc] for loc, ix in loc_ixs.items()}
#         )
#
#
# def _modified_abs(x: torch.Tensor):
#     epsilon = 1e-5  # Add small epsilon to gradient to avoid getting stuck at 0
#     # return F.relu(x) + epsilon * (x - x.detach().clone())
#     return torch.abs(x) + epsilon * (x - x.detach().clone())
