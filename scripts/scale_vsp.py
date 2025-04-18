import openvsp as vsp
import numpy as np
from scipy.optimize import minimize

import os

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import (
    AerostructGeometry,
    AerostructPoint,
)
import openmdao.api as om
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta
from openaerostruct.geometry.utils import generate_vsp_surfaces


# Load the VSP file
# vsp_file = "BWB_3.vsp3"


# Scale the model
def scale_vsp(vsp_file, scale_factor) -> str:
    """Scale the VSP model by a given factor. Scaling the entire geometry uniformly.

    Args:
        vsp_file (_type_): Name of the VSP file to be scaled. Include the file directory if necessary.
        scale_factor (_type_): Scale Factor, not percentage. For example, 0.5 will scale the model by 50%.

    Returns:
        _type_: Name of the scaled VSP file. Include the file directory if necessary.

    Example:
        >>>  new_file_name = scale_vsp(vsp_file="BWB_3.vsp3", scale_factor=0.5)
    """

    # Load the VSP file
    vsp.ReadVSPFile(vsp_file)
    # Scale the model
    vsp.ScaleSet(1, scale_factor)
    # Save the scaled model to a new file
    percentage = round(scale_factor * 100, 1)  # round to the nearest integer
    percentage_string = str(percentage).replace(".", "_")
    scaled_file_name = vsp_file[:-5] + f"_{percentage_string}percent_scaled.vsp3"
    vsp.WriteVSPFile(scaled_file_name)
    print(f"Scaled VSP file saved to: {scaled_file_name}")
    vsp.ClearVSPModel()
    return scaled_file_name


# %%


def get_air_density(altitude_m) -> float:
    """
    Estimates air density based on altitude using the U.S. Standard Atmosphere.

    Args:
    altitude_m : float
        Altitude in meters.

    Returns:
    rho : float
        Air density in kg/m^3.
    """
    # Constants
    temp_sea_level = 288.15  # sea level standard temperature (K)
    lapse_rate = 0.0065  # temperature lapse rate (K/m)
    pressure_sea_level = 101325  # sea level standard pressure (Pa)
    air_gas_constant = 287.053  # specific gas constant for dry air (J/(kg*K))
    gravity = 9.80665  # earth surface gravitational acceleration (m/s^2)

    # Temperature at altitude
    temp = temp_sea_level - lapse_rate * altitude_m

    # Pressure at altitude
    pressure = pressure_sea_level * (temp / temp_sea_level) ** (
        gravity / (lapse_rate * air_gas_constant)
    )

    # Density at altitude
    rho = pressure / (air_gas_constant * temp)

    return rho


def calc_mach_number(speed, speed_of_sound) -> float:
    """Calculates the Mach number as a function of speed and speed of sound."""
    return speed / speed_of_sound


def get_speed_of_sound(altitude: float) -> float:
    """
    Calculates the speed of sound for a given altitude using the International Standard Atmosphere (ISA) model.
    Args:
        altitude (float): Altitude in meters.
    Returns:
        float: Speed of sound in meters per second.
    """
    temperature = 288.15 - 6.5 * altitude / 1000  # ISA temperature lapse rate
    pressure = (
        101325.0 * (1 - 0.00002265 * altitude) ** 5.25588
    )  # ISA pressure altitude formula
    rho = pressure / (287.05 * temperature)  # ISA density formula
    return (1.4 * 287.05 * (temperature)) ** 0.5


def get_air_viscosity(temp) -> float:
    """
    Calculates the dynamic viscosity of air at a given temperature. Using Sutherland's law.
     Args:
        temp (float): Temperature in Celsius.
     Returns:
        float: Dynamic viscosity of air in kg/(m*s).
    """
    A = 1.458e-6
    B = 110.4
    C = 0.702
    T = temp + B
    mu = A * T ** (3 / 2) / (T + C)
    return mu


def air_temperature(altitude) -> float:
    """
    Calculates the air temperature as a function of altitude.
     Args:
        altitude (float): Altitude in meters.
     Returns:
        float: Temperature of air in Celsius.
    """
    temp = 15.0  # at sea level
    temp -= 0.0065 * altitude / 1000.0  # subtract temperature decrease due to altitude
    return temp


def calc_reynolds_number(rho: float, v, L, mu) -> float:
    """
    Calculates the Reynolds number for a given fluid with the specified
    density, velocity, length, and dynamic viscosity.
     Args:
        rho (float): Density of the fluid in kg/m^3.
        v (float): Velocity of the fluid in m/s.
        L (float): Length of a characteristic dimension of the flow in meters.
        mu (float): Dynamic viscosity of the fluid in Pa*s.
     Returns:
        float: Reynolds number for the flow.
    """
    return (rho * v * L) / mu


def calc_froude_number(air_speed: float, L: float = 1) -> float:
    """
    Calculates the Froude number for a given air speed and length.
     Args:
        air_speed (float): The air speed of the object.
        L (float, optional): The length of the object. Defaults to 1.
     Returns:
        float: The calculated Froude number.
    """
    g = 9.81
    return air_speed / np.sqrt(g * L)


# %% Create the input vectors
alt = 10000
rho = get_air_density(alt)
speed_of_sound = get_speed_of_sound(alt)
mach = 0.84
air_speed = mach * speed_of_sound
air_viscosity = get_air_viscosity(air_temperature(alt))
baseline_reynolds = calc_reynolds_number(rho, air_speed, 1.0, air_viscosity)


# Define your OpenAeroStruct function here
def OpenAeroStruct(scale, alpha, mach, altitude, youngs_modulus):
    # Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
    # These should be for an airfoil with the chord scaled to 1.
    # We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
    # We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
    # The first and last x-coordinates of the upper and lower surfaces must be the same

    upper_x = np.arange(0.1, 0.61, 0.01, dtype="complex128")
    lower_x = upper_x
    upper_y = np.array(
        [
            0.0447,
            0.046,
            0.0472,
            0.0484,
            0.0495,
            0.0505,
            0.0514,
            0.0523,
            0.0531,
            0.0538,
            0.0545,
            0.0551,
            0.0557,
            0.0563,
            0.0568,
            0.0573,
            0.0577,
            0.0581,
            0.0585,
            0.0588,
            0.0591,
            0.0593,
            0.0595,
            0.0597,
            0.0599,
            0.06,
            0.0601,
            0.0602,
            0.0602,
            0.0602,
            0.0602,
            0.0602,
            0.0601,
            0.06,
            0.0599,
            0.0598,
            0.0596,
            0.0594,
            0.0592,
            0.0589,
            0.0586,
            0.0583,
            0.058,
            0.0576,
            0.0572,
            0.0568,
            0.0563,
            0.0558,
            0.0553,
            0.0547,
            0.0541,
        ],
        dtype="complex128",
    )  # noqa: E201, E241
    lower_y = np.array(
        [
            -0.0447,
            -0.046,
            -0.0473,
            -0.0485,
            -0.0496,
            -0.0506,
            -0.0515,
            -0.0524,
            -0.0532,
            -0.054,
            -0.0547,
            -0.0554,
            -0.056,
            -0.0565,
            -0.057,
            -0.0575,
            -0.0579,
            -0.0583,
            -0.0586,
            -0.0589,
            -0.0592,
            -0.0594,
            -0.0595,
            -0.0596,
            -0.0597,
            -0.0598,
            -0.0598,
            -0.0598,
            -0.0598,
            -0.0597,
            -0.0596,
            -0.0594,
            -0.0592,
            -0.0589,
            -0.0586,
            -0.0582,
            -0.0578,
            -0.0573,
            -0.0567,
            -0.0561,
            -0.0554,
            -0.0546,
            -0.0538,
            -0.0529,
            -0.0519,
            -0.0509,
            -0.0497,
            -0.0485,
            -0.0472,
            -0.0458,
            -0.0444,
        ],
        dtype="complex128",
    )
    # Create a dictionary to store options about the surface
    scaled_file_name = scale_vsp("BWB_6.vsp3", scale)
    # vsp_file = os.path.join(os.path.dirname(__file__), "BWB_3.vsp3")
    vsp_file = os.path.join(os.path.dirname(__file__), scaled_file_name)
    # vsp_file = os.path.join(os.path.dirname(__file__), "NASA_N2A_Hybrid_Wing_Body.vsp3")
    # mesh_dict = {"num_y": 5, "num_x": 5, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 7}

    # mesh, twist_cp = generate_mesh(mesh_dict)

    surfaces = generate_vsp_surfaces(
        vsp_file, symmetry=False, include=["BWB"]  # "Tail" "BWB"
    )
    # print(surfaces)

    np.save("surfaces.npy", surfaces)

    youngs_modulus = youngs_modulus * 73.1e9  # [Pa] Young's modulus
    reynolds_0 = 6395780.1
    altitude = altitude * 10000  # [m] Altitude in meters
    alt = altitude
    rho = get_air_density(alt)
    rho_M = get_air_density(10000)
    speed_of_sound = get_speed_of_sound(alt)
    air_speed = mach * speed_of_sound
    air_viscosity = get_air_viscosity(air_temperature(alt))
    reynolds = calc_reynolds_number(rho, air_speed, scale, air_viscosity)

    # Create a dictionary to store options about the surface
    # mesh_dict = {
    #    "num_y": 51,
    #    "num_x": 7,
    #    "wing_type": "uCRM_based",
    #    "symmetry": True,
    #    "chord_cos_spacing": 0,
    #    "span_cos_spacing": 0,
    #    "num_twist_cp": 6,
    # }

    # mesh, twist_cp = generate_mesh(mesh_dict)

    twist_cp = np.array([9, 10, 12, 10, 8, 6])
    spar_thickness_cp = np.array([0.05, 0.01, 0.01, 0.006, 0.008, 0.007])
    skin_thickness_cp = np.array([0.05, 0.015, 0.02, 0.015, 0.01, 0.005])
    t_over_c_cp = np.array([0.075, 0.1, 0.125, 0.16, 0.125, 0.08])

    surf_dict = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": False,  # if true, model one half of wing
        "S_ref_type": "wetted",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        # "mesh": mesh,
        "twist_cp": np.append(twist_cp, np.flip(twist_cp)),
        "fem_model_type": "wingbox",
        "data_x_upper": upper_x,
        "data_x_lower": lower_x,
        "data_y_upper": upper_y,
        "data_y_lower": lower_y,
        "spar_thickness_cp": np.append(
            spar_thickness_cp, np.flip(spar_thickness_cp)
        ),  # [m]
        "skin_thickness_cp": np.append(skin_thickness_cp, np.flip(skin_thickness_cp)),
        "original_wingbox_airfoil_t_over_c": 0.12,
        # Aerodynamic deltas.
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        # They can be used to account for things that are not included, such as contributions from the fuselage, nacelles, tail surfaces, etc.
        "CL0": 0.07,
        "CD0": 0.002,
        "with_viscous": True,  # if true, compute viscous drag
        "with_wave": False,  # if true, compute wave drag
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.05,  # percentage of chord with laminar
        # flow, used for viscous drag
        "c_max_t": 0.38,  # chordwise location of maximum thickness
        "t_over_c_cp": np.append(t_over_c_cp, np.flip(t_over_c_cp)),
        # Structural values are based on aluminum 7075
        "E": youngs_modulus,  # [Pa] Young's modulus
        "G": (
            youngs_modulus / 2 / 1.33
        ),  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
        "yield": (420.0e6 / 1.5),  # [Pa] allowable yield stress
        "mrho": 2.78e3,  # [kg/m^3] material density
        "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
        "wing_weight_ratio": 1.25,
        "exact_failure_constraint": False,  # if false, use KS function
        "struct_weight_relief": False,
        "distributed_fuel_weight": True,
        "fuel_density": 803.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
        "Wf_reserve": 15000.0 * (rho / rho_M) * (scale**3),  # [kg] reserve fuel mass
    }

    # Update each surface with default options
    for surface in surfaces:
        surface.update(surf_dict)

    # surfaces = [surf_dict]

    # Create the problem and assign the model group
    prob = om.Problem()
    # Add problem information as an independent variables component
    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output(
        "v", val=np.array([air_speed, 0.64 * 340.294]), units="m/s"
    )
    indep_var_comp.add_output("alpha", val=alpha, units="deg")
    indep_var_comp.add_output("alpha_maneuver", val=0.0, units="deg")
    indep_var_comp.add_output("Mach_number", val=np.array([mach, 0.64]))
    indep_var_comp.add_output(
        "re",
        val=np.array(
            [
                reynolds,
                1.225 * 340.294 * 0.64 * 1.0 / (1.81206 * 1e-5),
            ]
        ),
        units="1/m",
    )  # 0.348 * 295.07 * mach * 1.0 / (1.43 * 1e-5)
    indep_var_comp.add_output("rho", val=np.array([rho, 1.225]), units="kg/m**3")
    indep_var_comp.add_output("CT", val=0.53 / 3600, units="1/s")
    indep_var_comp.add_output("R", val=14.307e6, units="m")
    indep_var_comp.add_output(
        "W0",
        val=(148000 + surf_dict["Wf_reserve"]) * (rho / rho_M) * (scale**3),
        units="kg",
    )
    indep_var_comp.add_output(
        "speed_of_sound", val=np.array([295.07, 340.294]), units="m/s"
    )
    indep_var_comp.add_output("load_factor", val=np.array([1.0, 2.25]))
    indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")
    indep_var_comp.add_output(
        "fuel_mass", val=10000.0 * (rho / rho_M) * (scale**3), units="kg"
    )

    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

    # Loop over each surface in the surfaces list
    for surface in surfaces:
        # Get the surface name and create a group to contain components
        # only for this surface
        name = surface["name"]

        aerostruct_group = AerostructGeometry(surface=surface)

        # Add group to the problem with the name of the surface.
        prob.model.add_subsystem(name, aerostruct_group)

    # Loop through and add a certain number of aerostruct points
    for i in range(2):
        point_name = "AS_point_{}".format(i)
        # Connect the parameters within the model for each aerostruct point

        # Create the aero point group and add it to the model
        AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)

        prob.model.add_subsystem(point_name, AS_point)

        # Connect flow properties to the analysis point
        prob.model.connect("v", point_name + ".v", src_indices=[i])
        prob.model.connect("Mach_number", point_name + ".Mach_number", src_indices=[i])
        prob.model.connect("re", point_name + ".re", src_indices=[i])
        prob.model.connect("rho", point_name + ".rho", src_indices=[i])
        prob.model.connect("CT", point_name + ".CT")
        prob.model.connect("R", point_name + ".R")
        prob.model.connect("W0", point_name + ".W0")
        prob.model.connect(
            "speed_of_sound", point_name + ".speed_of_sound", src_indices=[i]
        )
        prob.model.connect("empty_cg", point_name + ".empty_cg")
        prob.model.connect("load_factor", point_name + ".load_factor", src_indices=[i])
        prob.model.connect("fuel_mass", point_name + ".total_perf.L_equals_W.fuelburn")
        prob.model.connect("fuel_mass", point_name + ".total_perf.CG.fuelburn")

        for surface in surfaces:
            name = surface["name"]

            if surf_dict["distributed_fuel_weight"]:
                prob.model.connect(
                    "load_factor", point_name + ".coupled.load_factor", src_indices=[i]
                )

            com_name = point_name + "." + name + "_perf."
            prob.model.connect(
                name + ".local_stiff_transformed",
                point_name + ".coupled." + name + ".local_stiff_transformed",
            )
            prob.model.connect(
                name + ".nodes", point_name + ".coupled." + name + ".nodes"
            )

            # Connect aerodyamic mesh to coupled group mesh
            prob.model.connect(
                name + ".mesh", point_name + ".coupled." + name + ".mesh"
            )
            if surf_dict["struct_weight_relief"]:
                prob.model.connect(
                    name + ".element_mass",
                    point_name + ".coupled." + name + ".element_mass",
                )

            # Connect performance calculation variables
            prob.model.connect(name + ".nodes", com_name + "nodes")
            prob.model.connect(
                name + ".cg_location",
                point_name + "." + "total_perf." + name + "_cg_location",
            )
            prob.model.connect(
                name + ".structural_mass",
                point_name + "." + "total_perf." + name + "_structural_mass",
            )

            # Connect wingbox properties to von Mises stress calcs
            prob.model.connect(name + ".Qz", com_name + "Qz")
            prob.model.connect(name + ".J", com_name + "J")
            prob.model.connect(name + ".A_enc", com_name + "A_enc")
            prob.model.connect(name + ".htop", com_name + "htop")
            prob.model.connect(name + ".hbottom", com_name + "hbottom")
            prob.model.connect(name + ".hfront", com_name + "hfront")
            prob.model.connect(name + ".hrear", com_name + "hrear")

            prob.model.connect(name + ".spar_thickness", com_name + "spar_thickness")
            prob.model.connect(name + ".t_over_c", com_name + "t_over_c")

    prob.model.connect("alpha", "AS_point_0" + ".alpha")
    prob.model.connect("alpha_maneuver", "AS_point_1" + ".alpha")

    # Here we add the fuel volume constraint componenet to the model
    prob.model.add_subsystem("fuel_vol_delta", WingboxFuelVolDelta(surface=surface))
    prob.model.connect("wing.struct_setup.fuel_vols", "fuel_vol_delta.fuel_vols")
    prob.model.connect("AS_point_0.fuelburn", "fuel_vol_delta.fuelburn")

    if surf_dict["distributed_fuel_weight"]:
        prob.model.connect(
            "wing.struct_setup.fuel_vols",
            "AS_point_0.coupled.wing.struct_states.fuel_vols",
        )
        prob.model.connect(
            "fuel_mass", "AS_point_0.coupled.wing.struct_states.fuel_mass"
        )

        prob.model.connect(
            "wing.struct_setup.fuel_vols",
            "AS_point_1.coupled.wing.struct_states.fuel_vols",
        )
        prob.model.connect(
            "fuel_mass", "AS_point_1.coupled.wing.struct_states.fuel_mass"
        )

    comp = om.ExecComp("fuel_diff = (fuel_mass - fuelburn) / fuelburn", units="kg")
    prob.model.add_subsystem(
        "fuel_diff", comp, promotes_inputs=["fuel_mass"], promotes_outputs=["fuel_diff"]
    )
    prob.model.connect("AS_point_0.fuelburn", "fuel_diff.fuelburn")

    ## Use these settings if you do not have pyOptSparse or SNOPT
    prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-6
    # prob.driver.opt_settings['Major iterations limit'] = 5

    # # The following are the optimizer settings used for the EngOpt conference paper
    # # Uncomment them if you can use SNOPT
    # prob.driver = om.pyOptSparseDriver()
    # prob.driver.options['optimizer'] = "SNOPT"
    # prob.driver.opt_settings['Major optimality tolerance'] = 5e-6
    # prob.driver.opt_settings['Major feasibility tolerance'] = 1e-8
    prob.driver.opt_settings["Major iterations limit"] = 40

    # recorder = om.SqliteRecorder("aerostruct_test.db")
    # prob.driver.add_recorder(recorder)

    # We could also just use prob.driver.recording_options['includes']=['*'] here, but for large meshes the database file becomes extremely large. So we just select the variables we need.
    prob.driver.recording_options["includes"] = [
        "alpha",
        "rho",
        "v",
        "cg",
        "AS_point_1.cg",
        "AS_point_0.cg",
        "AS_point_0.coupled.wing_loads.loads",
        "AS_point_1.coupled.wing_loads.loads",
        "AS_point_0.coupled.wing.normals",
        "AS_point_1.coupled.wing.normals",
        "AS_point_0.coupled.wing.widths",
        "AS_point_1.coupled.wing.widths",
        "AS_point_0.coupled.aero_states.wing_sec_forces",
        "AS_point_1.coupled.aero_states.wing_sec_forces",
        "AS_point_0.wing_perf.CL1",
        "AS_point_1.wing_perf.CL1",
        "AS_point_0.coupled.wing.S_ref",
        "AS_point_1.coupled.wing.S_ref",
        "wing.geometry.twist",
        "wing.mesh",
        "wing.skin_thickness",
        "wing.spar_thickness",
        "wing.t_over_c",
        "wing.structural_mass",
        "AS_point_0.wing_perf.vonmises",
        "AS_point_1.wing_perf.vonmises",
        "AS_point_0.coupled.wing.def_mesh",
        "AS_point_1.coupled.wing.def_mesh",
    ]

    prob.driver.recording_options["record_objectives"] = True
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_inputs"] = True

    prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)

    prob.model.add_design_var("wing.twist_cp", lower=-15.0, upper=15.0, scaler=0.1)
    prob.model.add_design_var(
        "wing.spar_thickness_cp", lower=0.005, upper=0.02, scaler=1e2
    )
    prob.model.add_design_var(
        "wing.skin_thickness_cp", lower=0.005, upper=0.02, scaler=1e2
    )
    prob.model.add_design_var(
        "wing.geometry.t_over_c_cp", lower=0.07, upper=0.16, scaler=10.0
    )
    # prob.model.add_design_var("fuel_mass", lower=0.0, upper=2e5, scaler=1e-5)
    prob.model.add_design_var("alpha_maneuver", lower=-15.0, upper=15)
    prob.model.add_design_var("Mach_number", lower=0.4, upper=1, scaler=1e-10)

    # prob.model.add_constraint("AS_point_0.CL", equals=0.82)
    prob.model.add_constraint("AS_point_1.L_equals_W", equals=0.0)
    prob.model.add_constraint("AS_point_1.wing_perf.failure", upper=0.0)

    prob.model.add_constraint("fuel_vol_delta.fuel_vol_delta", lower=0.0)
    prob.model.add_constraint("fuel_diff", equals=0.0)

    # Set up the problem
    prob.setup()

    # om.view_model(prob)

    # prob.check_partials(form='central', compact_print=True)

    try:
        # prob.run_model()  # fuel 94400, wingbox mass 23427

        prob.run_driver()  # fuel 90509, wingbox mass 21174

        print("The fuel burn value is", prob["AS_point_0.fuelburn"][0], "[kg]")
        print(
            "The wingbox mass (excluding the wing_weight_ratio) is",
            prob["wing.structural_mass"][0] / surf_dict["wing_weight_ratio"],
            "[kg]",
        )

        print("CL:", prob["AS_point_0.CL"][0])
        print("CD:", prob["AS_point_0.CD"][0])
        print("CL/CD:", prob["AS_point_0.CL"][0] / prob["AS_point_0.CD"][0])

        clcd = prob["AS_point_0.CL"][0] / prob["AS_point_0.CD"][0]

        return clcd
    except:
        print("An exception occurred, returning CL/CD = 1000")
        return 1000


# Define your cost function here
def cost_function(x):
    scale, alpha, mach, altitude, youngs_modulus = x
    print(
        f"scale: {scale}, alpha: {alpha}, mach: {mach}, altitude: {altitude}, Young's Modulus: {youngs_modulus*73.1e9}"
    )
    # Get result from OpenAeroStruct
    result = OpenAeroStruct(scale, alpha, mach, altitude, youngs_modulus)

    rho = get_air_density(altitude)
    speed_of_sound = get_speed_of_sound(alt)
    air_speed = mach * speed_of_sound
    air_viscosity = get_air_viscosity(air_temperature(alt))
    reynolds = calc_reynolds_number(rho, air_speed, scale, air_viscosity)

    # Compare result with full-scale result and return the difference
    full_scale_result = 23.17  # TODO: change this to the full scale result

    # Calculate the cost function
    clcd_cost = abs(result - full_scale_result) ** 2
    reynolds_cost = (
        30 * abs((reynolds - baseline_reynolds) / baseline_reynolds) ** 2
    )  # 30 looks good for scaling the penalty
    mach_cost = 3000 * abs((mach - 0.84) / 0.84) ** 2

    print(
        f"CL/CD cost: {clcd_cost}, Reynolds cost: {reynolds_cost}, Mach cost: {mach_cost}"
    )

    return clcd_cost + reynolds_cost + mach_cost


def callback(xk):
    # Custom callback function
    print(f"Iteration {callback.iteration}: x = {xk}")
    callback.iteration += 1


callback.iteration = 1  # Initialize the iteration count


# Define your constraints here
constraints = (
    {"type": "ineq", "fun": lambda x: 0.2 - x[0]},  # scale should be less than 0.25
    {"type": "ineq", "fun": lambda x: x[0] - 0.01},  # scale should be greater than 0.01
    {
        "type": "ineq",
        "fun": lambda x: x[1] - 0.0,
    },  # alpha should be greater than 0
    {"type": "ineq", "fun": lambda x: 10 - x[1]},  # Alpha should be less than 10
    {
        "type": "ineq",
        "fun": lambda x: 0.87 - x[2],
    },  # Mach number should be less than 0.87
    {"type": "ineq", "fun": lambda x: x[3] - 0},  # altitude greater than 0
    {"type": "ineq", "fun": lambda x: 1.2 - x[3]},  # altitude less than 15km
    {
        "type": "ineq",
        "fun": lambda x: x[2] - 0.77,
    },  # Mach number should be greater than 0.77
    {
        "type": "ineq",
        "fun": lambda x: 3 - x[4],
    },  # Young's modulus should not be greater than 3 times original value
    {
        "type": "ineq",
        "fun": lambda x: x[4] - 0.1,
    },
)  # Froude number should be greater than 0.1

# Define your initial guess for the scale, alpha, Mach, and Froude numbers,Young's modulus scaling factor
initial_guess = [0.1, 0, 0.84, 1, 1]

# Perform optimization using SLSQP
result = minimize(
    cost_function,
    initial_guess,
    method="SLSQP",
    constraints=constraints,
    callback=callback,
    # options={"eps": 1e-1},
)
