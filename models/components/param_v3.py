import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO -- save parameters to df


class ParameterModel(nn.Module):

    def get_parameters(self, xs: dict) -> torch.Tensor:
        raise NotImplementedError


class FixedParameterModel(ParameterModel):
    """
    Map the same constant value to each location
    """

    def __init__(self, c: float):
        super().__init__()
        self._c = nn.Parameter(torch.Tensor(c), requires_grad=False)

    def get_parameters(self, xs: dict) -> torch.Tensor:
        return torch.ones(len(xs['location'])).unsqueeze(-1).to(self._c.device) * self._c


class GroupedParameterMapping(ParameterModel):
    """
    Map location groups to a value
    """

    def __init__(self,
                 location_groups: dict,
                 init_val: float = 0.0,
                 init_val_group: dict = None,
                 ):
        """

        """
        super().__init__()
        # Make sure all group are assigned an initial value
        assert init_val_group is None or all(
            [group in init_val_group.keys() for group in set(location_groups.values())])

        n_groups = len(set(location_groups.values()))

        # Rename all groups to some index
        group_to_ix = {
            g: i for i, g in enumerate(set(location_groups.values()))
        }

        # Map all locations to these indices
        self._loc_ixs = {
            loc: group_to_ix[g] for loc, g in location_groups.items()
        }

        if init_val_group is None:
            self._ps = nn.ParameterList(
                [torch.tensor(float(init_val), requires_grad=True) for _ in range(n_groups)]
            )
        else:
            self._ps = nn.ParameterList(
                [torch.tensor(float(init_val_group[g]), requires_grad=True) for g in group_to_ix.keys()]
            )

    def get_parameters(self, xs: dict):
        return self(xs)

    @property
    def locations(self):
        return list(self._loc_ixs.keys())

    @property
    def location_group_ixs(self) -> dict:
        return dict(self._loc_ixs)

    @property
    def location_values(self) -> dict:
        return {
            loc: self._ps[ix].item() for loc, ix in self.location_group_ixs.items()
        }

    def forward(self, xs: dict):
        locations = xs['location']

        ps = torch.cat([self._ps[self._loc_ixs[loc]].view(1, 1) for loc in locations],
                       dim=0,
                       )

        return ps

    def as_ungrouped(self) -> 'GroupedParameterMapping':
        """
        Recreate the current parameter mapping without grouped locations
        That is, each location is assigned one value (the one of the group its currently assigned to)
        :return: a new ParameterMapping object without location groups
        """
        # Assign an index (group) to all locations individually
        loc_ixs = {loc: ix for ix, loc in enumerate(self.locations)}
        # Get the values that are currently assigned to all locations
        loc_vals = self.location_values
        # Create a new parameter mapping without location grouping
        return GroupedParameterMapping(
            location_groups=loc_ixs,
            init_val_group={ix: loc_vals[loc] for loc, ix in loc_ixs.items()}
        )


class LocalParameterMapping(GroupedParameterMapping):

    def __init__(self,
                 locations: list,
                 init_val: float = 0.0,
                 init_val_group: dict = None,
                 ):
        super().__init__(
            {loc: i for i, loc in enumerate(locations)},
            init_val,
            init_val_group,
        )


class GlobalParameterMapping(GroupedParameterMapping):

    def __init__(self,
                 locations: list,
                 init_val: float = 0.0,
                 init_val_group: dict = None,
                 ):
        super().__init__(
            {loc: 0 for loc in locations},
            init_val,
            init_val_group,
        )


"""
    Utility functions
"""


def _modified_abs(x: torch.Tensor):
    epsilon = 1e-5  # Add small epsilon to gradient to avoid getting stuck at 0
    return torch.abs(x) + epsilon * (x - x.detach().clone())
