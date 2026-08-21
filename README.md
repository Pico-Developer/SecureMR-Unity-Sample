# SpatialML-Unity-Samples

`SpatialML-Unity-Samples` is a Unity project containing PICO SpatialML examples for protected camera access, on-device machine learning, spatial mapping, and secure rendering.

## Requirements

- Unity `6000.3.13f1`
- Android Build Support for Unity
- A SpatialML-capable PICO headset, such as Project Swan
- The PICO Unity SDK dependency declared in [`Packages/manifest.json`](Packages/manifest.json)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Pico-Developer/SpatialML-Unity-Samples.git
   cd SpatialML-Unity-Samples
   ```
2. Open the project with Unity `6000.3.13f1`.
3. Switch the active build target to Android if necessary.
4. Open `Assets/Scenes/SceneLoader.unity` to browse the enabled samples, or open a sample scene directly.
5. Build and run the selected scene on the headset.

## Samples

| Sample           | Scene                                 | Description                                                                         |
| ---------------- | ------------------------------------- | ----------------------------------------------------------------------------------- |
| Scene Loader     | `Assets/Scenes/SceneLoader.unity`     | Runtime launcher for navigating between the sample scenes.                          |
| Color Picker     | `Assets/Scenes/ColorPickerDemo.unity` | Samples a color through the SpatialML pipeline and applies it to a rendered object. |
| Minimal App      | `Assets/Scenes/MinimalApp.unity`      | Small introductory example of creating and running a SpatialML pipeline.            |
| MNIST Wild       | `Assets/Scenes/MnistwildDemo.unity`   | Runs a LiteRT MNIST model on camera input and renders the classification result.    |
| UFO              | `Assets/Scenes/UFODemo.unity`         | Uses LiteRT face detection and depth estimation to place a 3D UFO in the scene.     |
| Stylize          | `Assets/Scenes/StylizeDemo.unity`     | Runs LiteRT style prediction and transfer models and renders stylized portals.      |
| SpatialML Sample | `Assets/Scenes/SpatialMLSample.unity` | Demonstrates a basic SpatialML pipeline that renders a model and text.              |

## Project Layout

```text
Assets/
├── Gltf/            # 3D assets used by the samples
├── MLModels/        # LiteRT/TFLite models and associated metadata
├── Resources/       # PICO and runtime resources
├── Scenes/          # Launcher and sample scenes
├── Scripts/         # Sample behavior and navigation scripts
├── Settings/        # Universal Render Pipeline settings
├── Textures/        # Style-transfer source textures
└── XR/              # XR loader and project settings
Packages/            # Unity package dependencies
ProjectSettings/     # Unity project configuration
```

Unity model assets use the `.tflite.bytes` suffix so Unity imports them as `TextAsset` instances.

## Updating the PICO Unity SDK

The PICO Unity SDK is configured as the `com.bytedance.pico.xr` dependency in [`Packages/manifest.json`](Packages/manifest.json). See the [PICO Unity SDK import documentation](https://developer.picoxr.com/document/pico-unity-quickstart/import-the-sdk-and-select-a-rendering-mode/) before changing the package revision.

## SpatialML documentation
For more information and to try out more advanced experiments, consult the [SpatialML Guide](https://developer.picoxr.com/document/pico-unity-spatialml/)

## License

See [`LICENSE`](LICENSE).
