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
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using Color = Unity.XR.PXR.SecureMR.Color;

public class SecureMRSample : MonoBehaviour
{
    int image_width = 3248;   // Same as VST_IMAGE_WIDTH
    int image_height = 2464;  // Same as VST_IMAGE_HEIGHT
    int crop_x1 = 1444;
    int crop_y1 = 1332;
    int crop_x2 = 2045;
    int crop_y2 = 1933;
    int crop_width = 224;
    int crop_height = 224;
    
    private Provider provider;
    private Pipeline pipeline;
    private Pipeline pipeline2;
    
    private Tensor debugGltf;
    private Tensor debugGltfPlaceholder;
    private Tensor debugGltfPlaceholder2;

    public TextAsset tvGltf;
    
    // Start is called before the first frame update
    void Awake()
    {
        PXR_Manager.EnableVideoSeeThrough = true;
        //string filepath = Path.Combine(Application.persistentDataPath, "mnist_md5_48fe972.bin");
        //byte[] fileBytes = File.ReadAllBytes(filepath);

        provider = new Provider(image_width, image_height);
        //CreatePipeline();
        CreateRender();
    }
    
    void CreatePipeline()
    {
        //create provider and pipeline
        pipeline2 = provider.CreatePipeline();

        //create operator
        var vstAccessOp = pipeline2.CreateOperator<RectifiedVstAccessOperator>();
        var getAffineOp = pipeline2.CreateOperator<GetAffineOperator>();
        var applyAffineOp = pipeline2.CreateOperator<ApplyAffineOperator>();

        //create tensor
        var rawRgb = pipeline2.CreateTensor<byte,Matrix>(3, new TensorShape(image_height, image_width));
        var cropRgbWrite = pipeline2.CreateTensorReference<byte,Matrix>(3, new TensorShape(crop_width, crop_height));
        var cropRgbGlobal = provider.CreateTensor<byte,Matrix>(3, new TensorShape(crop_width, crop_height));
        var affineMat = pipeline2.CreateTensor<float,Matrix>(1, new TensorShape(2,3));
        var srcPoints  = pipeline2.CreateTensor<float,Point>(2, new TensorShape(3));
        var dstPoints  = pipeline2.CreateTensor<float,Point>(2, new TensorShape(3));
        
        float[] srcPointsData = {crop_x1,crop_y1,crop_x2,crop_y1,crop_x2,crop_y2};
        float[] dstPointsData = {0,0,crop_width,0,crop_width,crop_height};
        
        srcPoints.Reset(srcPointsData);
        dstPoints.Reset(dstPointsData);
        
        //set operator input and output
        vstAccessOp.SetResult("left image", rawRgb);
        
        getAffineOp.SetOperand("src",srcPoints);
        getAffineOp.SetOperand("dst",dstPoints);
        getAffineOp.SetResult("result",affineMat);
        
        applyAffineOp.SetOperand("affine",affineMat);
        applyAffineOp.SetOperand("src image",rawRgb);
        applyAffineOp.SetResult("dst image",cropRgbWrite);

        var pipelineIOPair = pipeline2.CreateTensorMapping();
        pipelineIOPair.Set(cropRgbWrite,cropRgbGlobal);
        
        
        pipeline2.Execute(pipelineIOPair);
    }
    
    void RunPipeline()
    {
        
    }

    void CreateRender()
    {
        pipeline = provider.CreatePipeline();

        RenderTextOperatorConfiguration renderTextConfiguration = new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960);
        
        var renderTextOp = pipeline.CreateOperator<RenderTextOperator>(renderTextConfiguration);
        var renderGltfOp = pipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
        
        var text = pipeline.CreateTensor<sbyte,Scalar>(1, new TensorShape(30));
        var startPosition = pipeline.CreateTensor<float,Point>(2, new TensorShape(1));
        var colors = pipeline.CreateTensor<byte,Color>(4, new TensorShape(2));
        var textureId = pipeline.CreateTensor<ushort,Scalar>(1, new TensorShape(1));
        var fontSize = pipeline.CreateTensor<float,Scalar>(1, new TensorShape(1));
        var poseMat  = pipeline.CreateTensor<float,Matrix>(1, new TensorShape(4,4));
        debugGltfPlaceholder = pipeline.CreateTensorReference<Gltf>();
        
        renderTextOp.SetOperand("text",text);
        renderTextOp.SetOperand("start",startPosition);
        renderTextOp.SetOperand("colors",colors);
        renderTextOp.SetOperand("texture ID",textureId);
        renderTextOp.SetOperand("font size",fontSize);
        renderTextOp.SetOperand("gltf",debugGltfPlaceholder);
        
        renderGltfOp.SetOperand("gltf",debugGltfPlaceholder);
        renderGltfOp.SetOperand("world pose",poseMat);
        
        string textValue = "Hello World";
        text.Reset(Encoding.UTF8.GetBytes(textValue));
        
        float[] startValue = {0.1f, 0.3f};
        startPosition.Reset(startValue);
        
        byte[] colorsValue = {255, 255, 255, 255, 0, 0, 0, 255};
        colors.Reset(colorsValue);
        
        int[] textureId_value = {0};
        textureId.Reset(textureId_value);
        
        float[] fontSize_value = {144.0f};
        fontSize.Reset(fontSize_value);

        float[] poseMatValue = {0.5f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.5f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.5f, -1.5f,
            0.0f, 0.0f, 0.0f, 1.0f};
        poseMat.Reset(poseMatValue);

        var gltfData = tvGltf.bytes;
        debugGltf = provider.CreateTensor<Gltf>(gltfData);

        /*//create operator
        var loadTextureOp = pipeline.CreateOperator<LoadTextureOperator>();
        var updateGltfOp = pipeline.CreateOperator<UpdateGltfOperator>();
        var renderGltfOp2 = pipeline.CreateOperator<SwitchGltfRenderStatusOperator>();

        //create tensor
        var gltfMaterialIndex = pipeline.CreateTensor<ushort,Scalar>(1, new TensorShape(1));
        var gltfTextureIndex = pipeline.CreateTensor<ushort,Scalar>(1, new TensorShape(1));
        var poseMat2 = pipeline.CreateTensor<float,Matrix>(1, new TensorShape(4,4));
        var debugGltfPlaceholder2 = pipeline.CreateTensorReference<Gltf>();
        var cropRgbRead = pipeline.CreateTensor<byte,Matrix>(3, new TensorShape(crop_width, crop_height));

        //set operator input and output
        loadTextureOp.SetOperand("rgb image", cropRgbRead);
        loadTextureOp.SetOperand("gltf", debugGltfPlaceholder2);
        loadTextureOp.SetResult("texture ID", gltfTextureIndex);

        updateGltfOp.SetOperand("gltf",debugGltfPlaceholder2);
        updateGltfOp.SetOperand("material ID",gltfMaterialIndex);
        updateGltfOp.SetOperand("value",gltfTextureIndex);

        renderGltfOp2.SetOperand("gltf",debugGltfPlaceholder2);
        renderGltfOp2.SetOperand("world pose",poseMat2);

        float[] poseMatValue2 =
            {0.5f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.5f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.5f, -1.5f,
            0.0f, 0.0f, 0.0f, 1.0f};
        poseMat2.Reset(poseMatValue2);*/

    }

    private void Update()
    {
        RenderFrame();
    }
    
    void RenderFrame()
    {
        var pipelineIOPair = pipeline.CreateTensorMapping();
        pipelineIOPair.Set(debugGltfPlaceholder,debugGltf);
        pipeline.Execute(pipelineIOPair);
    }
}
