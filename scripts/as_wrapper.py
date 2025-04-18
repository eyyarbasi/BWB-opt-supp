import openvsp as vsp
from math import pi


def reynolds_number(
    altitude: float, temperature: float, velocity: float, ref_length=1
) -> float:
    """
    Calculates Reynolds number as a function of altitude, temperature, velocity, and length.
    Parameters:
        altitude (float): Altitude in meters
        temperature (float): Temperature in Celsius
        velocity (float): Velocity of fluid in m/s
        ref_length (float): Length of the object in meters
        !the length is set to 1.0 by default, but will change with the scale
    Returns:
        float: Reynolds number (dimensionless)
    """
    # Calculate air density and viscosity using functions defined previously
    rho = air_density(altitude, temperature)
    mu = air_viscosity(altitude, temperature)
    # Calculate Reynolds number using formula
    Re = rho * velocity * ref_length / mu
    print(f"Re = {Re}")
    return Re


def mach_number(altitude: float, temperature: float, velocity: float) -> float:
    """
    Calculates Mach number as a function of altitude, temperature, and velocity.
    Parameters:
        altitude (float): Altitude in meters
        temperature (float): Temperature in Celsius
        velocity (float): Velocity in m/s
    Returns:
        float: Mach number (dimensionless)
    """
    # Calculate the speed of sound using the temperature at given altitude
    c = speed_of_sound(temperature, altitude)
    # Calculate the Mach number
    M = velocity / c
    print(f"M = {M}")
    return M


def air_density(altitude: float, temperature: float) -> float:
    """
    Calculates air density as a function of altitude and temperature.
    Parameters:
    altitude (float): Altitude in meters
    temperature (float): Temperature in degrees Celsius
    Returns:
    float: Air density in kg/m^3
    """
    # Calculate temperature in Kelvin
    T = temperature + 273.15
    # Calculate standard atmospheric pressure and temperature at sea level
    P0 = 101325  # N/m^2
    T0 = 288.15  # K
    # Calculate lapse rate and gas constant for air
    L = -0.0065  # K/meters
    R = 8.31447 / 0.0289644  # J/(mol*K)
    # Calculate pressure and density using barometric formula
    g0 = 9.80665  # m/s^2
    P = P0 * (1 + L * altitude / T0) ** (-g0 / (L * R))
    rho = P / (R * T)
    print(f"rho = {rho}" + " kg/m^3")
    return rho


def speed_of_sound(temperature: float, altitude: float) -> float:
    """
    Calculate the speed of sound based on temperature and altitude.

    Parameters:
    temperature (float): Temperature in degrees Celsius.
    altitude (float): Altitude in meters.

    Returns:
    float: Speed of sound in meters per second.
    """
    # Calculate the temperature in Kelvin
    T = temperature + 273.15

    # Calculate the pressure at the given altitude using the barometric formula
    P = 101325 * ((1 - 2.25577e-5 * altitude) ** 5.25588)

    # Calculate the speed of sound using the ideal gas law
    gamma = 1.4  # Specific heat ratio for air
    R = 287.058  # Gas constant for air
    c = (gamma * R * T) ** 0.5
    print(f"c = {c} m/s")

    return c * (1 - 0.0001 * altitude)


def air_viscosity(altitude: float, temperature: float) -> float:
    """
    Calculate the dynamic viscosity of air as a function of altitude and
    temperature, using Sutherland's law.
    Parameters:
    altitude (float): Altitude in meters.
    temperature (float): Temperature in Celsius.
    Returns:
    float: Viscosity of air in Pa s.
    """
    # Calculate the temperature in Kelvin
    T = temperature + 273.15
    # Calculate the Sutherland constant
    S = 110.4
    # Boltzmann constant
    k_B = 1.380e-23
    # Calculate the reference temperature and viscosity
    T_ref = 291.15
    mu_ref = 1.789e-5
    # Calculate the density of air
    rho = air_density(altitude, temperature)
    # Calculate the mean free path of air
    sigma = 3.65e-10  # collision diameter of air molecules in meters
    lambda_ = (9.0 / 16.0) * ((k_B * T) / (pi * sigma**2 * rho))
    # Calculate the dynamic viscosity of air
    mu = (
        mu_ref
        * ((T_ref + S) / (T + S))
        * ((T / T_ref) ** 1.5)
        * ((lambda_ + S) / (lambda_ + T))
    )
    print(f"mu = {mu}")
    return mu


def speed_from_mach_number(
    mach_number: float, altitude: float, temperature: float
) -> float:
    """
    Calculates speed in m/s from Mach number, altitude, and temperature.
    Parameters:
        mach_number (float): Mach number (dimensionless)
        altitude (float): Altitude in meters
        temperature (float): Temperature in Celsius
    Returns:
        float: Speed in m/s
    """
    # Calculate the speed of sound using the temperature at given altitude
    c = speed_of_sound(temperature, altitude)
    # Calculate the speed
    speed = mach_number * c
    # Adjust for the effect of altitude on air density
    speed *= air_density(altitude, temperature) / air_density(0, 15)

    return speed


# Load the VSP file
# vsp_file = "BWB_3.vsp3"


# Scale the model
def scale_vsp(vsp_file, scale_factor):
    """Scale the VSP model by a given factor. Scaling the entire geometry uniformly.

        Args:
            vsp_file (_type_): Name of the VSP file to be scaled. Include the file directory if necessary.
    1
        Returns:
            _type_: Name of the scaled VSP file. Include the file directory if necessary.

        Example:
            >>>  new_file_name = scale_vsp(vsp_file="BWB_3.vsp3", scale_factor=0.5)
    """

    # vsp.GetParmVal("WOLNFQQZUZK")

    # Load the VSP file
    vsp.ReadVSPFile(vsp_file)
    # Scale the model
    vsp.ScaleSet(1, scale_factor)
    # Save the scaled model to a new file
    percentage = scale_factor * 100

    # Get the root chord of the BWB = fuslage length
    root_chord = vsp.GetParmVal("NJOFORJMOL", "Chord", "XSecCurve")
    percentage_string = str(percentage).replace(".", "_")
    scaled_file_name = vsp_file[:-5] + f"_{percentage_string}percent_scaled.vsp3"
    vsp.WriteVSPFile(scaled_file_name)
    print(f"Scaled VSP file saved to: {scaled_file_name}")
    return scaled_file_name, root_chord


def main():
    # Scaling the model by a factor
    # TODO: Compute the factor based on similitudes.

    scale_factor = 1.0  # initiate with scale_factor =1
    vsp_file = "BWB_3.vsp3"

    inputs = {"altitude": 0, "temperature": 15, "velocity": 100}

    scaled_file, root_chord = scale_vsp(vsp_file=vsp_file, scale_factor=scale_factor)
    # Calculate Mach number
    mach = mach_number(
        altitude=inputs["altitude"],
        temperature=inputs["temperature"],
        velocity=inputs["velocity"],
    )
    # Calculate Reynolds number
    re = reynolds_number(
        altitude=inputs["altitude"],
        temperature=inputs["temperature"],
        velocity=inputs["velocity"],
        ref_length=root_chord,
    )

    # Size dependent variables
    # Wf_reserve
    # reference length for Reynolds number
    # reference_length = fuselage_length * scale_factor

    # Parameters to modify: E,G, yield, mrho, fuel_density,
    # Mach_number
    # v is derived from v = M*a
    # re is derived from re = rho*v*L/mu where v = Mach_number*a and L=1


if __name__ == "__main__":
    main()  # Run the main function
