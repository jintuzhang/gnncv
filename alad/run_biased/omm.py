import sys
from mdtraj.reporters import DCDReporter

import openmm
import openmmplumed
from openmm import app
from openmm import unit
from openmmtools import integrators


gro_file = '../data/A.gro'
top_file = '../data/topol.top'

gro = app.GromacsGroFile(gro_file)
top = app.GromacsTopFile(
        top_file,
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir='../data'
)
system = top.createSystem(
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds
)
with open('plumed.inp', 'r') as fp:
    plumed = openmmplumed.PlumedForce(fp.read())
system.addForce(plumed)

integrator = integrators.GeodesicBAOABIntegrator(
    K_r=2,
    temperature=300 * unit.kelvin,
    collision_rate=1 / unit.picosecond,
    timestep=0.002 * unit.picoseconds
)

platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
simulation = app.Simulation(
    top.topology, system, integrator, platform, properties
)

simulation.context.setPositions(gro.positions)
dcd_reporter = DCDReporter('traj.dcd', reportInterval=500)
screen_reporter = app.StateDataReporter(
    sys.stdout, 50000, step=True, potentialEnergy=True, temperature=True
)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(screen_reporter)

simulation.step(int(1.5E7))
