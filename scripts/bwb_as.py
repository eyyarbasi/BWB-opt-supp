"""Aerostructural optimization with Tubular Spar"""

import numpy as np
import os

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import (
    AerostructGeometry,
    AerostructPoint,
)
from openaerostruct.geometry.geometry_group import Geometry
import openmdao.api as om
from openaerostruct.utils.constants import grav_constant
from openaerostruct.geometry.utils import generate_vsp_surfaces

# Create a dictionary to store options about the surface
vsp_file = os.path.join(os.path.dirname(__file__), "BWB_3.vsp3")
# mesh_dict = {"num_y": 5, "num_x": 5, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 7}

# mesh, twist_cp = generate_mesh(mesh_dict)

surfaces = generate_vsp_surfaces(vsp_file, symmetry=True, include=["BWB"])  # "Tail"

twist_cp = np.array([9, 10, 12, 10, 8, 6])
spar_thickness_cp = np.array([0.05, 0.01, 0.01, 0.006, 0.008, 0.007])
skin_thickness_cp = np.array([0.05, 0.015, 0.02, 0.015, 0.01, 0.005])
t_over_c_cp = np.array([0.075, 0.1, 0.125, 0.16, 0.125, 0.08])

surf_options = {
    # Wing definition
    # "name": "wing",  # name of the surface
    "symmetry": True,  # if true, model one half of wing
    # reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "fem_model_type": "tube",
    "fem_origin": 0.35,  # normalized chordwise location of the spar
    "thickness_cp": np.array([0.1, 0.2, 0.3]),
    "twist_cp": np.zeros(3),
    # "mesh": mesh,
    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    "CL0": 0.0,  # CL of the surface at alpha=0
    "CD0": 0.015,  # CD of the surface at alpha=0
    # Airfoil properties for viscous drag calculation
    "k_lam": 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
    "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    "with_viscous": True,
    "with_wave": False,  # if true, compute wave drag
    # Structural values are based on aluminum 7075
    "E": 70.0e9,  # [Pa] Young's modulus of the spar
    "G": 30.0e9,  # [Pa] shear modulus of the spar
    "yield": 500.0e6 / 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
    "mrho": 3.0e3,  # [kg/m^3] material density
    "fem_origin": 0.35,  # normalized chordwise location of the spar
    "wing_weight_ratio": 2.0,
    "struct_weight_relief": False,  # True to add the weight of the structure to the loads on the structure
    "distributed_fuel_weight": False,
    # Constraints
    "exact_failure_constraint": False,  # if false, use KS function
}


# Update each surface with default options
for surface in surfaces:
    surface.update(surf_options)

# Create the problem and assign the model group
prob = om.Problem()

MACH = 0.64
V_SOUND = 340.294  # ["m/s"]
RHO = 1.225  # ["kg/m**3"]
V = MACH * V_SOUND
LOAD_FACTOR = 1

# Add problem information as an independent variables component
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=V, units="m/s")
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("Mach_number", val=MACH)
indep_var_comp.add_output("re", val=RHO * V * 1.0 / (1.81206 * 1e-5), units="1/m")
indep_var_comp.add_output("rho", val=RHO, units="kg/m**3")
indep_var_comp.add_output(
    "CT", val=grav_constant * 17.0e-6, units="1/s"
)  # speciific fuel consumption
indep_var_comp.add_output("R", val=11.165e6, units="m")
indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")
indep_var_comp.add_output("beta", val=0.0, units="deg")  # Sideslip angle
indep_var_comp.add_output("speed_of_sound", val=V_SOUND, units="m/s")
indep_var_comp.add_output("load_factor", val=LOAD_FACTOR)
indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# Add geometry group to the problem and add wing surface as a sub group.
# These groups are responsible for manipulating the geometry of the mesh,
# in this case spanwise twist.
geom_group = om.Group()
aerostruct_group = AerostructGeometry(surface=surface)
for surface in surfaces:
    geom_group.add_subsystem(surface["name"], Geometry(surface=surface))

prob.model.add_subsystem("geom", geom_group, promotes=["*"])
prob.model.add_subsystem("as", aerostruct_group)

# Create the aero point group and add it to the model
AS_point = AerostructPoint(surfaces=surfaces, rotational=True, compressible=True)
point_name = "flight_condition_0"

AS_point.set_input_defaults("beta", units="deg")


prob.model.add_subsystem(
    point_name,
    AS_point,
    promotes_inputs=[
        "v",
        "alpha",
        "Mach_number",
        "re",
        "rho",
        "CT",
        "R",
        "W0",
        "speed_of_sound",
        "empty_cg",
        "load_factor",
    ],
)

for surface in surfaces:
    name = surface["name"]
    # prob.model.connect(name + ".mesh", point_name + "." + name + ".def_mesh")

    # Perform the connections with the modified names within the
    # 'aero_states' group.
    prob.model.connect(
        name + ".mesh", point_name + ".coupled." + name + "." + "def_mesh"
    )

    com_name = point_name + "." + name + "_perf"
    prob.model.connect(
        "as" + ".local_stiff_transformed",
        point_name + ".coupled." + name + ".local_stiff_transformed",
    )
    prob.model.connect("as" + ".nodes", point_name + ".coupled." + name + ".nodes")

    # Connect aerodyamic mesh to coupled group mesh
    # prob.model.connect("as" + ".mesh", point_name + ".coupled." + name + ".mesh")

    # Connect performance calculation variables
    prob.model.connect("as" + ".radius", com_name + ".radius")
    prob.model.connect("as" + ".thickness", com_name + ".thickness")
    prob.model.connect("as" + ".nodes", com_name + ".nodes")
    prob.model.connect(
        "as" + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location"
    )
    prob.model.connect(
        "as" + ".structural_mass",
        point_name + "." + "total_perf." + name + "_structural_mass",
    )
    prob.model.connect("as" + ".t_over_c", com_name + ".t_over_c")

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder("aerostruct_bwb.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options["record_derivatives"] = True
prob.driver.recording_options["includes"] = ["*"]

# Setup problem and add design variables, constraint, and objective
prob.model.add_design_var("wing.twist_cp", lower=-10.0, upper=15.0)
prob.model.add_design_var("wing.thickness_cp", lower=0.01, upper=0.5, scaler=1e2)
prob.model.add_constraint("AS_point_0.wing_perf.failure", upper=0.0)
prob.model.add_constraint("AS_point_0.wing_perf.thickness_intersects", upper=0.0)

# Add design variables, constraisnt, and objective on the problem
prob.model.add_design_var("alpha", lower=-10.0, upper=10.0)
prob.model.add_constraint("AS_point_0.L_equals_W", equals=0.0)
# prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)
prob.model.add_objective("wing.structural_mass", scaler=1e-5)

# Set up the problem
prob.setup(check=True)

# Only run analysis
# prob.run_model()

# Run optimization
prob.run_driver()

print()
print("CL:", prob["AS_point_0.wing_perf.CL"])
print("CD:", prob["AS_point_0.wing_perf.CD"])
print("Fuel Burn: ", prob["AS_point_0.fuelburn"])
print("Structural Mass: ", prob["wing.structural_mass"])


# from openmdao.api import n2
# n2(prob)
