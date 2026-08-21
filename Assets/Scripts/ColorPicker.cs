// Copyright (2025) Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using ByteDance.PICO.XR;
using ByteDance.PICO.SecureMR;
using UnityEngine;
using Color = ByteDance.PICO.SecureMR.Color;

namespace PicoXR.SpatialML.Demo
{
    public class ColorPicker : MonoBehaviour, ISpatialMLServiceCleanup
    {
        public TextAsset cubeGltfAsset;
        public Transform cubePosition;
        public float pickColorEveryXSeconds = 0.033f;

        // Distance to push the cube along the controller's forward axis so it
        // appears past the end of the controller and does not overlap with it.
        [Tooltip("Distance (meters) to offset the cube along the controller's forward axis so it appears past the controller tip.")]
        public float cubeForwardOffset = 0.15f;

        // Rendering camera used to project the cube's world position into VST UV space.
        // Falls back to Camera.main if not assigned.
        [Tooltip("Camera used to approximate the VST left-eye view for UV projection. Defaults to Camera.main.")]
        public Camera projectionCamera;

        private Provider provider;
        private Pipeline pipeline;
        private Pipeline rendererPipeline;
        private Pipeline readerPipeline;

        private int vstWidth = 256;
        private int vstHeight = 256;

        private float elapsed = 0.0f;
        //Color Tensors
        public Tensor colorPlaceholder;
        public Tensor colorGlobal;
        public Tensor poseMatPlaceholder;

        public Tensor pickedColorPlaceholder;

        // Slice tensors for per-frame UV sampling
        private Tensor sliceGlobal;
        private Tensor slicePlaceholder;

        // GLTF tensors
        private Tensor gltfTensor;
        private Tensor gltfPlaceholder;
        private Tensor poseMatGlobal;

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        // Start is called once before the first execution of Update after the MonoBehaviour is created
        private void Start()
        {
            CreateProvider();
            CreatePipeline();
            CreateRenderer();
        }

        private void OnDestroy()
        {
            ShutdownSpatialML();
        }

        public void ShutdownSpatialML()
        {
            if (provider == null)
            {
                return;
            }

            Provider providerToDestroy = provider;
            provider = null;
            providerToDestroy.Destroy();
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreatePipeline()
        {
            if (provider == null)
            {
                Debug.LogError("Provider is not initialized.");
                return;
            }

            //create globals
            colorGlobal =
                provider.CreateTensor<float, Color>(4, new TensorShape(new[] { 1 }), new float[] { 1.0f, 0.0f, 0.0f, 1.0f });

            poseMatGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, -0.5f,
                    0.0f, 0.5f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            // Slice global describing the (x_start, x_end, y_start, y_end) pixel window
            // sampled from the rectified VST image. Updated each frame from the cube's
            // projected screen position so the picked color is the color *behind* the cube.
            int initX = vstWidth / 2;
            int initY = vstHeight / 2;
            sliceGlobal = provider.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }),
                new int[] { initX, initX + 1, initY, initY + 1 });
        }
        
        private void CreateRenderer()
        {
            if (provider == null)
            {
                Debug.LogError("Provider is not initialized.");
                return;
            }

            Debug.Log("Creating renderer...");

            // Create renderer pipeline
            rendererPipeline = provider.CreatePipeline();

            // Create operators
            var renderGltfOp = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var materialBaseColorConfig = new UpdateGltfOperatorConfiguration(SecureMRGltfOperatorAttribute.MaterialBaseColorFactor);
            var updateGltfOp = rendererPipeline.CreateOperator<UpdateGltfOperator>(materialBaseColorConfig);

            
            // Create tensors
            var gltfMaterialIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });

            // Create placeholder tensors
            gltfPlaceholder = rendererPipeline.CreateTensorReference<Gltf>();
            colorPlaceholder = rendererPipeline.CreateTensorReference<float, Color>(4, new TensorShape(new[] { 1 }));
            poseMatPlaceholder =
                rendererPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
             
            // Create global GLTF tensors
            gltfTensor = provider.CreateTensor<Gltf>(cubeGltfAsset.bytes);

            // Connect the operators
            renderGltfOp.SetOperand("gltf", gltfPlaceholder);
            renderGltfOp.SetOperand("world pose", poseMatPlaceholder);
 
            updateGltfOp.SetOperand("gltf", gltfPlaceholder);
            updateGltfOp.SetOperand("material ID", gltfMaterialIndex);
            updateGltfOp.SetOperand("value", colorPlaceholder);

            Debug.Log("Renderer created successfully.");
            
            //Create readerPipeline
            readerPipeline = provider.CreatePipeline();
            
            var vstOp = readerPipeline.CreateOperator<RectifiedVstAccessOperator>();
            var assignmentOperator2 = readerPipeline.CreateOperator<AssignmentOperator>();
            var assignmentOperator3 = readerPipeline.CreateOperator<AssignmentOperator>();
            var rgbToRgbaOp =
                readerPipeline.CreateOperator<ConvertColorOperator>(
                    new ColorConvertOperatorConfiguration(0)); // 0 = RGB to RGBA

            var uint8ToFloat32Op = readerPipeline.CreateOperator<AssignmentOperator>();
            var normalizeOp =
                readerPipeline.CreateOperator<ArithmeticComposeOperator>(
                    new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            

            var rawRgb = readerPipeline.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));

            // Slice placeholder - the actual sampling window is provided each frame
            // by binding sliceGlobal (which is reset based on the cube's projected UV).
            slicePlaceholder = readerPipeline.CreateTensorReference<int, Slice>(2, new TensorShape(new[] { 2 }));

            var pickedColor = readerPipeline.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { 1, 1 }));
            
            var pickedColorA = readerPipeline.CreateTensor<byte, Matrix>(4, new TensorShape(new[] { 1, 1 }));

            var pickedColorFloat =  readerPipeline.CreateTensor<float, Matrix>(4, new TensorShape(new[] { 1, 1 }));
            pickedColorPlaceholder = readerPipeline.CreateTensorReference<float, Color>(4, new TensorShape(new[] { 1 }));
            
            // Connect the operators
            vstOp.SetResult("left image", rawRgb);

            assignmentOperator2.SetOperand("src", rawRgb);
            assignmentOperator2.SetOperand("src slices", slicePlaceholder);
            assignmentOperator2.SetResult("dst", pickedColor);

            rgbToRgbaOp.SetOperand("src", pickedColor);
            rgbToRgbaOp.SetResult("dst", pickedColorA);
            
            uint8ToFloat32Op.SetOperand("src", pickedColorA);
            uint8ToFloat32Op.SetResult("dst", pickedColorFloat);

            normalizeOp.SetOperand("{0}", pickedColorFloat);
            normalizeOp.SetResult("result", pickedColorFloat);
            
            assignmentOperator3.SetOperand("src", pickedColorFloat);
            assignmentOperator3.SetResult("dst", pickedColorPlaceholder);
        }

        // Builds a Matrix4x4 representing the cube's world pose, offset along the
        // controller's forward axis by cubeForwardOffset so the cube renders past the
        // controller tip and does not overlap with the physical controller.
        private Matrix4x4 GetOffsetCubeMatrix()
        {
            var mat = cubePosition.localToWorldMatrix;
            // Column 2 of a TRS matrix is the local +Z (forward) axis in world space.
            Vector3 forward = new Vector3(mat.m02, mat.m12, mat.m22);
            Vector3 offset = forward * cubeForwardOffset;
            mat.m03 += offset.x;
            mat.m13 += offset.y;
            mat.m23 += offset.z;
            return mat;
        }

        private void UpdatePoseMatGlobal()
        {
            var mat = GetOffsetCubeMatrix();
            //rotations still seem reversed on y axis

            poseMatGlobal.Reset(new[]
            {
                mat.m00, mat.m01, mat.m02, mat.m03 ,
                mat.m10, mat.m11, mat.m12, mat.m13 - 1.36144f,
                -mat.m20, -mat.m21, -mat.m22, -mat.m23,
                mat.m30, mat.m31, mat.m32, mat.m33,
            });
        }

        // Projects the cube's offset world position into VST left-image UV pixel
        // coordinates, then resets sliceGlobal so the reader pipeline samples the
        // pixel directly behind the cube. Falls back to the image center if the
        // cube is behind the camera or off-screen.
        private void UpdateSliceGlobal()
        {
            Camera cam = projectionCamera != null ? projectionCamera : Camera.main;
            if (cam == null)
            {
                return;
            }

            Vector3 worldPos = GetOffsetCubeMatrix().GetColumn(3);
            Vector3 viewport = cam.WorldToViewportPoint(worldPos);

            // Behind camera - keep previous slice rather than sampling garbage.
            if (viewport.z <= 0f)
            {
                return;
            }

            // VST image origin is top-left, Unity viewport origin is bottom-left.
            int x = Mathf.Clamp((int)(viewport.x * vstWidth), 0, vstWidth - 2);
            int y = Mathf.Clamp((int)((1f - viewport.y) * vstHeight), 0, vstHeight - 2);

            sliceGlobal.Reset(new int[] { x, x + 1, y, y + 1 });
        }
                
        private ulong RenderFrame()
        {
            Debug.Log("Rendering frame...");

            UpdatePoseMatGlobal();

            var tensorMapping = new TensorMapping();
            tensorMapping.Set(gltfPlaceholder, gltfTensor);
            tensorMapping.Set(colorPlaceholder, colorGlobal);
            tensorMapping.Set(poseMatPlaceholder, poseMatGlobal);

            return rendererPipeline.Execute(tensorMapping);
        }

        private void PickColor(ulong pipelineRun)
        {
            UpdateSliceGlobal();

            var tensorMapping = new TensorMapping();
            tensorMapping.Set(slicePlaceholder, sliceGlobal);
            tensorMapping.Set(pickedColorPlaceholder, colorGlobal);

            readerPipeline.ExecuteAfter(pipelineRun, tensorMapping);
        }
        
        // Update is called once per frame
        private void Update()
        {
            if (provider == null)
            {
                return;
            }

            var pipelineRun = RenderFrame();
            elapsed += Time.deltaTime;
            if (elapsed > pickColorEveryXSeconds)
            {
                elapsed -= pickColorEveryXSeconds;
                PickColor(pipelineRun);
            }
            
        }

    }
}
