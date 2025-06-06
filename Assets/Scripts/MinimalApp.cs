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

using System;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;

namespace PicoXR.SecureMR.Demo
{
    public class MinimalApp : MonoBehaviour
    {
        public TextAsset helmetGltfAsset;
        public int vstWidth = 1024;
        public int vstHeight = 1024;

        private Provider provider;
        private Pipeline pipeline;
        private Tensor gltfTensor;
        private Tensor gltfPlaceholderTensor;
        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            CreateProvider();
            CreateGlobals();
            CreatePipeline();
        }

        private void Update()
        {
            RunPipeline();
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateGlobals()
        {
            // Create GLTF tensor
            gltfTensor = provider.CreateTensor<Gltf>(helmetGltfAsset.bytes);

        }

        private void CreatePipeline()
        {
            pipeline = provider.CreatePipeline();

            // Create transform matrix tensor
            int[] transformDim = { 4, 4 };
            var transformShape = new TensorShape(transformDim);
            float[] transformData = {
                0.5f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.5f, 0.0f, 0.25f,
                0.0f, 0.0f, 0.5f, -1.5f,
                0.0f, 0.0f, 0.0f, 1.0f
            };
            var poseTensor = pipeline.CreateTensor<float, Matrix>(1, transformShape, transformData);

            // Create GLTF tensor placeholder
            gltfPlaceholderTensor = pipeline.CreateTensorReference<Gltf>();

            // Create render GLTF operator
            var renderGltfOperator = pipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            renderGltfOperator.SetOperand("gltf", gltfPlaceholderTensor);
            renderGltfOperator.SetOperand("world pose", poseTensor);
        }
        
        private void RunPipeline()
        {
            Debug.Log("Running pipeline...");

            var tensorMapping = new TensorMapping();
          
            tensorMapping.Set(gltfPlaceholderTensor, gltfTensor);

            pipeline.Execute(tensorMapping);
        }

    }
}