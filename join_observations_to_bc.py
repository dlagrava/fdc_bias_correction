import argparse

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_bc_nc", help="netCDF file containing the simulations")
    parser.add_argument("observation_nc", help="netCDF file containing simulated and observed FDC")
    parser.add_argument("-output_dir", default=".")
    parser.add_argument("-output_file_name", default="bias_corrected_with_obs.nc")
    return parser.parse_args()


def main():
    args = parse_args()
    input_bc_nc = args.input_bc_nc
    observation_nc = args.observation_nc
    output_dir = args.output_dir
    output_file_name = args.output_file_name

    input_bc = xr.open_dataset(input_bc_nc)
    observation = xr.open_dataset(observation_nc)

    bc_start_year = input_bc.time.dt.year.min()
    bc_end_year = input_bc.time.dt.year.max()
    bc_slice = slice('%i-01-01' % bc_start_year, '%i-12-31' % bc_end_year)
    print(bc_slice)
    reduced_observation = observation.sel(time=bc_slice)

    bc_reaches = input_bc.rchid.values

    obs_values = np.ones_like(input_bc.bias_corrected.values) * np.NAN

    for bc_rch_idx, bc_reach_id in enumerate(bc_reaches):
        print(bc_reach_id)
        try:
            obs_rch_idx = np.where(observation.station_rchid.values == bc_reach_id)[0][0]
        except IndexError:
            print("No reach {} in observations, ignoring".format(bc_reach_id))
            continue

        obs_values[:, bc_rch_idx] = reduced_observation.river_flow_rate[:, obs_rch_idx]

    input_bc["river_flow_rate"] = input_bc.bias_corrected.copy()
    input_bc["river_flow_rate"].attrs.update(description="average flow", standard_name="river_flow_rate",
                                             long_name="observed_streamflow")
    input_bc["river_flow_rate"][:, :] = obs_values

    input_bc.to_netcdf(output_file_name)


if __name__ == "__main__":
    main()
