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

using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using Color = Unity.XR.PXR.SecureMR.Color;

namespace PicoXR.SecureMR.Demo
{
    public class ColorPicker : MonoBehaviour
    {
        public TextAsset cubeGltfAsset;
        public Transform cubePosition;
        public float pickColorEveryXSeconds = 0.033f;
  
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
        
        
        // GLTF tensors
        private Tensor gltfTensor;
        private Tensor gltfPlaceholder;
        private Tensor poseMatGlobal;

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        // Start is called once before the first execution of Update after the MonoBehaviour is created
        void Start()
        {
            CreateProvider();
            CreatePipeline();
            CreateRenderer();
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
            
            // use controller position to get picked color
            var mat = cubePosition.localToWorldMatrix;

            Vector3 controllerForward = new Vector3(mat.m02, mat.m12, mat.m22).normalized;
            
            int controllerX = Mathf.Clamp((int)((controllerForward.x + 1) * 0.5f * vstWidth), 0, vstWidth-2);
            int controllerY = Mathf.Clamp((int)((controllerForward.y + 1) * 0.5f * vstHeight), 0, vstHeight-2);
            
            var slice = readerPipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }),
            new int[] { controllerX, controllerX + 1, controllerY, controllerY + 1 });
            var pickedColor = readerPipeline.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { 1, 1 }));
            
            var pickedColorA = readerPipeline.CreateTensor<byte, Matrix>(4, new TensorShape(new[] { 1, 1 }));

            var pickedColorFloat =  readerPipeline.CreateTensor<float, Matrix>(4, new TensorShape(new[] { 1, 1 }));
            pickedColorPlaceholder = readerPipeline.CreateTensorReference<float, Color>(4, new TensorShape(new[] { 1 }));
            
            // Connect the operators
            vstOp.SetResult("left image", rawRgb);

            assignmentOperator2.SetOperand("src", rawRgb);
            assignmentOperator2.SetOperand("src slices", slice);
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

        private void UpdatePoseMatGlobal()
        {
            var mat = cubePosition.localToWorldMatrix;
            //rotations still seem reversed on y axis
            
            poseMatGlobal.Reset(new[]
            {
                mat.m00, mat.m01, mat.m02, mat.m03 ,
                mat.m10, mat.m11, mat.m12, mat.m13 - 1.36144f,
                -mat.m20, -mat.m21, -mat.m22, -mat.m23,
                mat.m30, mat.m31, mat.m32, mat.m33,
            });
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
            // colorGlobal.Reset(new float[] { Random.value, Random.value, Random.value, 1.0f }); 
 
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(pickedColorPlaceholder, colorGlobal);
 
            readerPipeline.ExecuteAfter(pipelineRun, tensorMapping);
            

        }
        
        // Update is called once per frame
        void Update()
        {
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