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
using System.Text;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using Color = Unity.XR.PXR.SecureMR.Color;

namespace PicoXR.SecureMR.Demo
{
    public class SecureMRSample : MonoBehaviour
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

            // RenderText Op
            RenderTextOperatorConfiguration renderTextConfiguration = new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960);
            var renderTextOp = pipeline.CreateOperator<RenderTextOperator>(renderTextConfiguration);

            var text = pipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 30 }),
                Encoding.UTF8.GetBytes("Hello World"));
            var startPosition = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors = pipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 }); // white text, black background
            var textureId = pipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize = pipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
                new float[] { 144.0f });
            
            renderTextOp.SetOperand("text",text);
            renderTextOp.SetOperand("start",startPosition);
            renderTextOp.SetOperand("colors",colors);
            renderTextOp.SetOperand("texture ID",textureId);
            renderTextOp.SetOperand("font size",fontSize);
            renderTextOp.SetOperand("gltf",gltfPlaceholderTensor);
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