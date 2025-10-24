# Robust-Bluetooth-Localization-GMEE-EKF
MATLAB implementation of a robust Bluetooth (BLE) localization system using the Generalized Minimum Error Entropy Extended Kalman Filter (GMEE-EKF) with Generalized Gaussian Distribution (GGD) weighting for improved accuracy under mixed noise.
# Bluetooth Indoor Localization System (GMEE-AEKF)

Bluetooth localization using Gaussian Mixture Maximum Entropy Estimation. Way more accurate than traditional methods.

## Overview

This is a bluetooth indoor positioning system I built that uses RSSI signal strength for localization. The core algorithm is GMEE-AEKF, which adapts its parameters dynamically and handles noisy environments pretty well.

## Features

- Adaptive filtering: parameters adjust themselves based on error
- Gaussian mixture: multiple components handle nonlinearity
- Compared against LS method so you can see the improvement
- Anti-oscillation mechanism
- Damping control for stable convergence

## Requirements

MATLAB R2018b or newer. No toolboxes needed.

## Usage

Just run `bluetooth_localization_gmee_aekf.m` in MATLAB.

## Parameters

```matlab
n_components = 3        % Gaussian components
state_dim = 4          % [x, y, vx, vy]
dt = 0.1               % time step
tx_power = -59         % transmit power
noise_std = 3.0        % RSSI noise
path_loss_exp = 2.5    % path loss exponent
```

## Test Results

Circular trajectory test:

- GMEE-AEKF: ~0.5m average error
- LS method: ~1.2m average error

Pretty significant difference.

## File Structure

```
├── bluetooth_localization_gmee_aekf.m    main script
├── README.md                              this file
└── results/                               output plots
```

## Customization

### Change beacon positions

```matlab
beacons = [
    0, 0;
    10, 0;
    10, 10;
    0, 10
];
```

### Change trajectory

```matlab
% Circular
t = linspace(0, 4*pi, 200);
true_trajectory = [5 + 3*cos(t)', 5 + 3*sin(t)'];

% Linear
% true_trajectory = [linspace(2, 8, 200)', linspace(2, 8, 200)'];
```

## Core Algorithm

The adaptive parameter adjustment is key. Alpha controls the entropy estimation shape, beta controls error tolerance. The system automatically tunes both based on current error.

Covariance update uses Joseph form for stability:
```matlab
P = (I - K*H) * P * (I - K*H)' + K * R * K'
```

## Applications

- Indoor navigation
- Warehouse asset tracking
- Robot localization
- Personnel tracking systems

## License

MIT License

## Contact

Open an issue or reach out directly if you have questions.

---

Code is in the document above, just copy and run.
