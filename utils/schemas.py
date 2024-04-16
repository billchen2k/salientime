'''
schemas.py
Project: utils
Created: 2023-08-24 00:34:39
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2024-04-16 17:01:27
Modified By: Bill Chen (bill.chen@live.com)
'''
from apiflask import Schema
from apiflask import fields, validators
from numpy import require

# Shared configurations
available_datasets = ['global-hs', 'rdi']

# Shared params
field_dataset = fields.String(
    required=True,
    validate=validators.OneOf(available_datasets),
    description='Name of the dataset.'
)

field_geo_bound = fields.List(
    fields.List(fields.Float(), validate=validators.Length(equal=2)),
    validate=validators.Length(equal=2),
    required=False,
    description='''
Optional spatial boundary for the query. In the format of:

- `[[Longitude1, Latitude1], [Longitude2, Latitude2]]`.

Where the first coordinate being the **left upper** corner, the second coordinate being the **right lower** corner.
The array index calculating will be performed on the server.
'''
)
field_start = fields.Integer(
    required=True,
    description='The start frame **index** of the focus range. 0 indicates the beginning of the dataset.'
)
field_end = fields.Integer(
    required=True,
    description='The end frame **index** of the focus range. 0 indicates the beginning of the dataset.'
)


class IndexOut(Schema):
    message = fields.String()


class FindFrameIn(Schema):

    dataset = field_dataset
    start = field_start
    end = field_end
    k = fields.Integer(
        required=True,
        description='The number of time steps to select from a given range [start, end] (Including the end).'
    )
    alpha = fields.Float(
        required=False,
        load_default=1.0,
        description='The weight of the structural variation.'
    )
    beta = fields.Float(
        required=False,
        load_default=0.0,
        description='The weight of the statistical variation. Requires alpha + beta = 1.'
    )
    agg = fields.String(
        required=False,
        load_default='max',
        validate=validators.OneOf(['max', 'min', 'avg']),
        description='The aggregation operation for the statistical variation.'
    )
    geo_bound = field_geo_bound


class FindFrameOut(Schema):
    frames = fields.List(
        fields.Integer(),
        required=True,
        description='The indices of selected k salient time steps.'
    )


class TrendIn(Schema):
    dataset = field_dataset
    start = field_start
    end = field_end
    geo_bound = field_geo_bound
    step = fields.Integer(
        required=False,
        description='The step for getting trend. If not specified, will be calculted in the backend.'
    )
    latent_only = fields.Boolean(
        required=False,
        description='If only return the latent code.'
    )


class TrendOut(Schema):
    pairwise_cossim = fields.List(
        fields.List(fields.Float()),
        description='''
The pair-wise cosine similarity (Structural Variation) of the input range. Shape (end - start + 1, 2)

- To calculate the temporal trend visualization, use step wise value:
  - Temporal Structural Variation at time t =  abs(cossim[t] - cossim[t - 1]);

- To calculate the relative trend visualization, just use this matrix:
  - Relative Structural Variation at time t from k = cossim[k, t]
        '''
    )
    latent_2d = fields.List(
        fields.List(fields.Float(), validate=validators.Length(equal=2)),
        description='''
A list of latent codes (reduced to 2 dimensions) for the focus time range. Shape: (end - start + 1, 2).

latent_2d[t] is a coordinate (x, y) in the 2D latent space for the given time t, where x and y are normalized to [0, 1].
        '''
    )
    trend_max = fields.List(
        fields.Float(),
        description='''
The maximum **raw** value for the given spatial region for the temporal trend visualization. Length: end - start + 1

- To calculate the temporal trend visualization, use normalized value:
  - Temporal Trend of max at time t =  trend[k];

- To calculate the relative trend visualization, use diffs:
  - Relative Statistical Variation for max at time t from k = abs(trend[k] - trend[k - 1]);
        '''
    )
    trend_min = fields.List(
        fields.Float(),
        description='Same as `trend_max`, but provides data for min aggregation.'
    )
    trend_avg = fields.List(
        fields.Float(),
        description='Same as `trend_max`, but provides data for average aggregation.'
    )
    step = fields.Integer(
        description='The step between each two points. Will be >= 1 if the end - start + 1 is too large.'
    )


class RawDataIn(Schema):
    dataset = field_dataset
    frame_index = fields.Integer(
        required=True,
        description='The index of the frame.'
    )

class RawDataInJson(Schema):
    geo_bound = field_geo_bound

class DatasetInfoIn(Schema):
    dataset = field_dataset

class DatasetInfoOut(Schema):
    total_frames = fields.Integer(
        description='The total number of frames in the dataset.'
    )
    dates = fields.List(
        fields.DateTime(format='iso'),
        description='The date of each frame in the dataset.'
    )
    description = fields.String(
        description='Description text for the dataset.'
    )
    geo_bound = fields.List(
    fields.List(fields.Float(), validate=validators.Length(equal=2)),
    validate=validators.Length(equal=2),
    description='''
The geographical boundary for all frames in the dataset. In the format of:

- `[[Longitude1, Latitude1], [Longitude2, Latitude2]]`.

Where the first coordinate being the **left upper** corner, the second coordinate being the **right lower** corner.
'''
)
    vmax = fields.Float(
        description='The maximum value of the dataset. (For colormapping)'
    )
    vmin = fields.Float(
        description='The minimum value of the dataset. (For colormapping)'
    )
    units = fields.String(
        description='Name of the units'
    )
    units_long = fields.String(
        description='Long name of the units'
    )

class LoadIn(Schema):
    dataset = field_dataset

class LoadInJson(Schema):
    geo_bound = field_geo_bound

class LoadOut(Schema):
    message = fields.String()
    time_eplapsed = fields.Float()
