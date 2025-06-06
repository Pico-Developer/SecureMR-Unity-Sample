# SecureMR Unity Samples

This repository contains Unity-based sample projects demonstrating SecureMR functionality with XR and machine learning integration.

## Project Structure

```plaintext
├── Assets/                      # Unity project assets
│   ├── Gltf/                   # 3D model assets (helmet, cube, TV models)
│   ├── MLModels/               # Machine learning model files (MNIST)
│   ├── Resources/              # Unity resource assets and PXR settings
│   ├── Samples/                # XR Interaction Toolkit samples
│   ├── Scenes/                 # Unity scene files including:
│   │   ├── ColorPickerDemo     # Color selection demonstration
│   │   ├── MinimalApp          # Basic application example
│   │   ├── MnistwildDemo       # MNIST ML model demonstration
│   │   └── SecureMRSample      # Main SecureMR functionality demo
│   ├── Scripts/                # C# script files for demos
│   ├── Settings/               # Project configuration files
│   ├── XR/                     # XR settings and loaders
│   └── XRI/                    # XR Interaction settings
├── Packages/                   # Unity package dependencies
└── ProjectSettings/            # Unity project configuration
```

## Available Demos

1. **Color Picker Demo**
   - Scene: `Assets/Scenes/ColorPickerDemo.unity`
   - Demonstrates color selection functionality

2. **Minimal App**
   - Scene: `Assets/Scenes/MinimalApp.unity`
   - Basic implementation example

3. **MNIST Wild Demo**
   - Scene: `Assets/Scenes/MnistwildDemo.unity`
   - Demonstrates machine learning integration using MNIST dataset

4. **SecureMR Sample**
   - Scene: `Assets/Scenes/SecureMRSample.unity`
   - Showcases simple SecureMR features
5. **UFO Demo**
   - Scene: `Assets/Scenes/UFO.unity`
   - Showcases end-to-end SecureMR functionality with face detection and depth estimation for rendering 3D objects in space

## Requirements

- Unity (6000.0.39f1)
- SecureMR Unity SDK
- PICO 4 Ultra device

## Getting Started

1. Clone this repository
2. Open the project in Unity
3. Open one of the demo scenes from the `Assets/Scenes` folder
4. Configure your XR device settings in the `Assets/XR/Settings` folder
5. Run the scene

## Additional Resources

- 3D Models: Available in `Assets/Gltf/`
- ML Models: MNIST model available in `Assets/MLModels/`

## How to update SecureMR SDK
Please refer to https://developer.picoxr.com/document/unity/import-the-sdk/ for more details.