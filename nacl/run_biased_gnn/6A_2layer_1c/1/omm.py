import os
import sys
from mdtraj.reporters import DCDReporter

import openmm
import openmmplumed
from openmm import app
from openmm import unit
from openmmtools import integrators

import backup
backup.back_up('traj.dcd')

pdb_file = '../../../data/r.pdb'
psf_file = '../../../data/solvate.psf'

pdb = app.PDBFile(pdb_file)
psf = app.CharmmPsfFile(
    psf_file,
    periodicBoxVectors=pdb.topology.getPeriodicBoxVectors()
)
params = app.CharmmParameterSet(
    '../../../data/par_all27_prot_lipid.prm',
    '../../../data/top_all27_prot_lipid.rtf'
)
system = psf.createSystem(
    params,
    nonbondedMethod=app.PME,
    nonbondedCutoff=0.9*unit.nanometer,
    constraints=app.HBonds
)

if os.path.exists('restart.chk'):
    plumed_file = 'plumed.restart.inp'
else:
    plumed_file = 'plumed.inp'
with open(plumed_file, 'r') as fp:
    plumed = openmmplumed.PlumedForce(fp.read())
system.addForce(plumed)

integrator = integrators.GeodesicBAOABIntegrator(
    K_r=4,
    temperature=300 * unit.kelvin,
    collision_rate=1 / unit.picosecond,
    timestep=0.002 * unit.picoseconds
)

platform = openmm.Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
simulation = app.Simulation(
    psf.topology, system, integrator, platform, properties
)

simulation.context.setPositions(pdb.positions)
dcd_reporter = DCDReporter('traj.dcd', reportInterval=5000)
screen_reporter = app.StateDataReporter(
    sys.stdout, 50000, step=True, potentialEnergy=True, temperature=True
)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(screen_reporter)

if os.path.exists('restart.chk'):
    simulation.loadCheckpoint('restart.chk')
simulation.runForClockTime(86300 * unit.second, checkpointFile='restart.chk')
