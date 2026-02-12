3D Flag Simulation with Wind (Mass-Spring System)
Overview

This project implements a physically-based cloth simulation using a mass-spring system in 3D. The system includes structural springs, gravity, damping, and a wind model based on Perlin noise turbulence.

The simulation is integrated using a fourth-order Runge-Kutta (RK4) method.

Features

3D mass-spring cloth model

Structural vertical and horizontal springs

Gravity and damping forces

Wind force with turbulence (Perlin noise)

Texture mapping onto dynamic geometry

RK4 numerical integration

Physical Model

Each mass node is connected through springs with stiffness coefficients:

Vertical springs

Horizontal springs

Forces included:

Hooke's law

Gravity

Aerodynamic pressure

Tangential wind shear

Damping

Numerical Method

Time integration is performed using RK4 for improved stability over Euler integration.

Dependencies
numpy
matplotlib
perlin-noise


Install with:

pip install -r requirements.txt

Run
python main.py

Author

Esther Menéndez
BSc Physics – Computational Simulation Focus
