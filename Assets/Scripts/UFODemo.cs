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
using System.Threading;
using System.Collections.Generic;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using Unity.Jobs;
using Color = Unity.XR.PXR.SecureMR.Color;
using System.Collections.Concurrent;

namespace PicoXR.SecureMR.Demo
{
    public class UFODemo : MonoBehaviour
    {
        public TextAsset faceModel;
        public TextAsset ufoGltfAsset;
        public TextAsset anchorMatrixAsset;
        public int numFramesToRun = -1;
        public float intervalBetweenPipelineRuns = 0.033f;
        private int vstWidth = 256;
        private int vstHeight = 256;

        private Provider provider;
        private Pipeline vstPipeline;
        private Pipeline modelInferencePipeline;
        private Pipeline map2dTo3dPipeline;
        private Pipeline renderPipeline;

        // Global tensors for sharing data between pipelines
        private Tensor vstOutputLeftUint8Global;
        private Tensor vstOutputRightUint8Global;
        private Tensor vstOutputLeftFp32Global;
        private Tensor vstTimestampGlobal;
        private Tensor vstCameraMatrixGlobal;
        private Tensor vstImagePlaceholder;
        // private Tensor isUfoDetectedGlobal;
        private Tensor currentPositionGlobal;
        private Tensor previousPositionGlobal;
        private Tensor leftEyeUVGlobal;

        // Pipeline tensors for VST
        private Tensor vstOutputLeftUint8Placeholder;
        private Tensor vstOutputLeftUint8Placeholder1;
        private Tensor vstOutputRightUint8Placeholder;
        private Tensor vstOutputRightUint8Placeholder1;
        private Tensor vstOutputLeftFp32Placeholder;
        private Tensor vstTimestampPlaceholder;
        private Tensor vstTimestampPlaceholder1;
        private Tensor vstCameraMatrixPlaceholder;
        private Tensor vstCameraMatrixPlaceholder1;

        // Pipeline tensors for model inference
        private ModelOperatorConfiguration modelConfig;
        private byte[] anchorBytes;
        private Tensor leftEyeUVPlaceholder;

        // Pipeline tensors for 2D to 3D mapping
        private Tensor uvPlaceholder;
        private Tensor map2dTo3dPositionPlaceholder;

        // Pipeline tensors for rendering
        private Tensor currentPositionRead;
        private Tensor previousPositionRead;

        // GLTF tensors
        private Tensor gltfTensor;
        private Tensor gltfPlaceholder;

        private float elapsed = 0.0f;
        private float vstElapsed = 0f;
        private float modelElapsed = 0f;
        private float mapElapsed = 0f;
        private float renderElapsed = 0f;

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
            Application.targetFrameRate = 30;
        }

        private volatile bool pipelinesReady = false;
        private ConcurrentQueue<Action> mainThreadActions = new ConcurrentQueue<Action>();
        private volatile bool vstPipelineReady = false;
        private volatile bool modelInferencePipelineReady = false;
        private volatile bool map2dTo3dPipelineReady = false;
        private volatile bool renderPipelineReady = false;
        private object pipelineLock = new object();

        private void Start()
        {
            try
            {
                CreateProvider();
                vstPipeline = provider.CreatePipeline();
                modelInferencePipeline = provider.CreatePipeline();
                map2dTo3dPipeline = provider.CreatePipeline();
                renderPipeline = provider.CreatePipeline();

                CreateGlobalTensors();

                new Thread(() => 
                {
                    try 
                    {
                        CreateVstPipeline();
                        lock(pipelineLock) { vstPipelineReady = true; }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Compute thread failed: {e.Message}");
                    }
                }).Start();

                new Thread(() =>
                {
                    try
                    {
                        CreateModelInferencePipeline();
                        lock(pipelineLock) { modelInferencePipelineReady = true; }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Model inference thread failed: {e.Message}");
                    }
                }).Start();

                new Thread(() =>
                {
                    try
                    {
                        CreateMap2dTo3dPipeline();
                        lock(pipelineLock) { map2dTo3dPipelineReady = true; }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Map2dTo3d thread failed: {e.Message}");
                    }
                }).Start();

                new Thread(() => 
                {
                    try 
                    {
                        CreateRenderPipeline();
                        lock(pipelineLock) { renderPipelineReady = true; }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Render thread failed: {e.Message}");
                    }
                }).Start();


            }
            catch (Exception e)
            {
                Debug.LogError($"Provider creation failed: {e.Message}");
            }
        }

        private void Update()
        {
            // Check if all pipelines are ready before proceeding
            if (!vstPipelineReady || !modelInferencePipelineReady || !map2dTo3dPipelineReady || !renderPipelineReady)
                return;

            vstElapsed += Time.deltaTime;
            modelElapsed += Time.deltaTime;
            mapElapsed += Time.deltaTime;

            // Only execute when pipelines are ready and interval reached
            if (vstElapsed >= 0.33f)
            {
                RunVstPipeline();
                vstElapsed = 0f;
            }

            if (modelElapsed >= 0.5f)
            {
                RunModelInferencePipeline();
                modelElapsed = 0f;
            }

            if (mapElapsed >= 0.5f)
            {
                RunMap2dTo3dPipeline();
                mapElapsed = 0f;
            }

            RunRenderPipeline();
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateGlobalTensors()
        {
            // Create global tensors for VST
            vstOutputLeftUint8Global = provider.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            vstOutputRightUint8Global = provider.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            vstOutputLeftFp32Global = provider.CreateTensor<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            vstTimestampGlobal = provider.CreateTensor<int, TimeStamp>(4, new TensorShape(new[] { 1 }));
            vstCameraMatrixGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 3 }));

            // Create global tensors for UFO detection
            modelConfig = new ModelOperatorConfiguration(faceModel.bytes, SecureMRModelType.QnnContextBinary, "face");
            anchorBytes = anchorMatrixAsset.bytes;
            leftEyeUVGlobal = provider.CreateTensor<int, Point>(2, new TensorShape(new[] { 1 }));

            // Create global tensors for position tracking
            currentPositionGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                    new float[] { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
            previousPositionGlobal = provider.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                    new float[] { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});

            byte[] gltfBytes = ufoGltfAsset.bytes;
            gltfTensor = provider.CreateTensor<Gltf>(gltfBytes);
        }

        private void CreateVstPipeline()
        {
            lock(pipelineLock)
            {
                try
                {
                    // Create pipeline placeholders for global tensors
                    vstOutputLeftUint8Placeholder = vstPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
                    vstOutputRightUint8Placeholder = vstPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
                    vstOutputLeftFp32Placeholder = vstPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
                    vstTimestampPlaceholder = vstPipeline.CreateTensorReference<int, TimeStamp>(4, new TensorShape(new[] { 1 }));
                    vstCameraMatrixPlaceholder = vstPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { 3, 3 }));

                    // Create VST access operator and connect tensors
                    var vstOp = vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
                    vstOp.SetResult("left image", vstOutputLeftUint8Placeholder);
                    vstOp.SetResult("right image", vstOutputRightUint8Placeholder);
                    vstOp.SetResult("timestamp", vstTimestampPlaceholder);
                    vstOp.SetResult("camera matrix", vstCameraMatrixPlaceholder);

                    // Create assignment operator for uint8 to fp32 conversion
                    var assignmentOp = vstPipeline.CreateOperator<AssignmentOperator>();
                    assignmentOp.SetOperand("src", vstOutputLeftUint8Placeholder);
                    assignmentOp.SetResult("dst", vstOutputLeftFp32Placeholder);

                    // Create arithmetic operator for normalization
                    var arithmeticOp = vstPipeline.CreateOperator<ArithmeticComposeOperator>(new ArithmeticComposeOperatorConfiguration("({0} / 255.0)"));
                    arithmeticOp.SetOperand("{0}", vstOutputLeftFp32Placeholder);
                    arithmeticOp.SetResult("result", vstOutputLeftFp32Placeholder);
                }
                catch (Exception e)
                {
                    Debug.LogError($"VST pipeline creation failed: {e.Message}");
                    mainThreadActions.Enqueue(() => pipelinesReady = false);
                }
            }
        }

        private void CreateModelInferencePipeline()
        {
            lock(pipelineLock)
            {
                try
                {
                    // Create pipeline placeholders for global tensors
                    vstImagePlaceholder = modelInferencePipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
                    leftEyeUVPlaceholder = modelInferencePipeline.CreateTensorReference<int, Point>(2, new TensorShape(new[] { 1 }));
                    
                    // 1. model inference
                    var ufoAnchor = modelInferencePipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 896, 16 }));
                    var ufoScores = modelInferencePipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 896, 1}));

                    // var modelConfig = new ModelOperatorConfiguration(ufoModel.bytes, SecureMRModelType.QnnContextBinary, "face");
                    modelConfig.AddInputMapping("image", "image", SecureMRModelEncoding.Float32);
                    modelConfig.AddOutputMapping("box_coords", "box_coords", SecureMRModelEncoding.Float32);
                    modelConfig.AddOutputMapping("box_scores", "box_scores", SecureMRModelEncoding.Float32);
                    
                    var modelOp = modelInferencePipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
                    modelOp.SetOperand("image", vstImagePlaceholder);
                    modelOp.SetResult("box_coords", ufoAnchor);
                    modelOp.SetResult("box_scores", ufoScores);

                    // 2. apply anchor
                    // var anchorBytes = anchorMatrixAsset.bytes;
                    var anchorFloats = new float[anchorBytes.Length / sizeof(float)];
                    Buffer.BlockCopy(anchorBytes, 0, anchorFloats, 0, anchorBytes.Length);
                    var anchorMatTensor = modelInferencePipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 896, 4 }), anchorFloats);
                    var ufoLandmarks = modelInferencePipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 896, 4 }));
                
                    // Create slice tensors for landmarks
                    var srcSliceTensor = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }), new[] { 0, 896, 4, 8 });
                    var anchorToSliceOp = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    anchorToSliceOp.SetOperand("src", ufoAnchor);
                    anchorToSliceOp.SetOperand("src slices", srcSliceTensor);
                    anchorToSliceOp.SetResult("dst", ufoLandmarks);

                    // Process anchor matrix
                    var anchorMatFirstTwoCols = modelInferencePipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 896, 2 }));
                    var srcSliceFirstTwoColsTensor = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }), new[] { 0, 896, 0, 2 });

                    var sliceOpAnchorMatFirstTwoCols = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    sliceOpAnchorMatFirstTwoCols.SetOperand("src", anchorMatTensor);
                    sliceOpAnchorMatFirstTwoCols.SetOperand("src slices", srcSliceFirstTwoColsTensor);
                    sliceOpAnchorMatFirstTwoCols.SetResult("dst", anchorMatFirstTwoCols);

                    // Duplicate anchor matrix
                    var anchorMatDuplicated = modelInferencePipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 896, 4 }));
                    
                    // Copy first two columns
                    var assignOpAnchorMatDuplicated = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignOpAnchorMatDuplicated.SetOperand("src", anchorMatFirstTwoCols);
                    assignOpAnchorMatDuplicated.SetOperand("dst slices", srcSliceFirstTwoColsTensor);
                    assignOpAnchorMatDuplicated.SetResult("dst", anchorMatDuplicated);

                    // Copy last two columns
                    var srcSliceLastTwoColsTensor = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }), new[] { 0, 896, 2, 4 });
                    var assignOpAnchorMatDuplicated2 = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignOpAnchorMatDuplicated2.SetOperand("src", anchorMatTensor);
                    assignOpAnchorMatDuplicated2.SetOperand("dst slices", srcSliceLastTwoColsTensor);
                    assignOpAnchorMatDuplicated2.SetResult("dst", anchorMatDuplicated);

                    // Process landmarks
                    var landmarksArithmeticOp = modelInferencePipeline.CreateOperator<ArithmeticComposeOperator>(
                        new ArithmeticComposeOperatorConfiguration("({0} / 256.0 + {1}) * 256.0"));
                    landmarksArithmeticOp.SetOperand("{0}", ufoLandmarks);
                    landmarksArithmeticOp.SetOperand("{1}", anchorMatDuplicated);
                    landmarksArithmeticOp.SetResult("result", ufoLandmarks);

                    // 3. get best detection - argmax
                    var bestFaceIndex = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 1 }));
                    var bestFaceIndexMat = modelInferencePipeline.CreateTensor<int, Matrix>(1, new TensorShape(new[] { 1, 1 }));
                    var bestFaceIndexPlusOne = modelInferencePipeline.CreateTensor<int, Matrix>(1, new TensorShape(new[] { 1, 1 }));

                    // Find best detection using ArgMax
                    var argmaxOp = modelInferencePipeline.CreateOperator<ArgmaxOperator>();
                    argmaxOp.SetOperand("operand", ufoScores);
                    argmaxOp.SetResult("result", bestFaceIndex);

                    // Convert scalar to matrix
                    var assignmentOpBestFaceIndex = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignmentOpBestFaceIndex.SetOperand("src", bestFaceIndex);
                    assignmentOpBestFaceIndex.SetResult("dst", bestFaceIndexMat);

                    // Add 1 to index
                    var arithmeticOp = modelInferencePipeline.CreateOperator<ArithmeticComposeOperator>(
                        new ArithmeticComposeOperatorConfiguration("{0} + 1"));
                    arithmeticOp.SetOperand("{0}", bestFaceIndexMat);
                    arithmeticOp.SetResult("result", bestFaceIndexPlusOne);

                    // Create slice tensors for best face
                    var srcSlicesBestFace = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }), 
                        new[] { 0, 896, 0, 4 });
                    var dstSlicesBestFace = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 1 }), 
                        new[] { 0, 1});
                    var dstSlicesBestFacePlusOne = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 1 }), 
                        new[] { 1, 1});

                    // Copy best face landmark
                    var assignmentOp1 = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignmentOp1.SetOperand("src", bestFaceIndexMat);
                    assignmentOp1.SetOperand("dst channel slice", dstSlicesBestFace);
                    assignmentOp1.SetResult("dst", srcSlicesBestFace);

                    var assignmentOp2 = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignmentOp2.SetOperand("src", bestFaceIndexPlusOne);
                    assignmentOp2.SetOperand("dst channel slice", dstSlicesBestFacePlusOne);
                    assignmentOp2.SetResult("dst", srcSlicesBestFace);

                    var bestFaceLandmark = modelInferencePipeline.CreateTensor<int, Matrix>(1, new TensorShape(new[] { 1, 4 }));
                    var assignmentOpBestFaceAnchors = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignmentOpBestFaceAnchors.SetOperand("src", ufoLandmarks);
                    assignmentOpBestFaceAnchors.SetOperand("src slices", srcSlicesBestFace);
                    assignmentOpBestFaceAnchors.SetResult("dst", bestFaceLandmark);

                    // Create best face landmark int32 tensor and operator
                    var bestFaceLandmarkInt32 = modelInferencePipeline.CreateTensor<int, Matrix>(1, new TensorShape(new[] { 1, 4 }));
                    var assignmentOpBestFaceLandmarkInt32 = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignmentOpBestFaceLandmarkInt32.SetOperand("src", bestFaceLandmark);
                    assignmentOpBestFaceLandmarkInt32.SetResult("dst", bestFaceLandmarkInt32);

                    // Extract left eye UV coordinates
                    var srcSliceLeftEyeUV = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }), new[] { 0, 1, 0, 2 });
                    // var dstChannelSliceLeftEyeUV = modelInferencePipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 1 }), new[] { 0, 2 });
                    var assignmentOpLeftEyeUV = modelInferencePipeline.CreateOperator<AssignmentOperator>();
                    assignmentOpLeftEyeUV.SetOperand("src", bestFaceLandmarkInt32);
                    assignmentOpLeftEyeUV.SetOperand("src slices", srcSliceLeftEyeUV);
                    // assignmentOpLeftEyeUV.SetOperand("dst channel slice", dstChannelSliceLeftEyeUV);
                    assignmentOpLeftEyeUV.SetResult("dst", leftEyeUVPlaceholder);
                }
                catch (Exception e)
                {
                    Debug.LogError($"Model inference pipeline creation failed: {e.Message}");
                    mainThreadActions.Enqueue(() => pipelinesReady = false);
                }
            }
        }

        private void CreateMap2dTo3dPipeline()
        {
            try
            {
                map2dTo3dPipeline = provider.CreatePipeline();

                // Step 1: Create pipeline placeholders
                uvPlaceholder = map2dTo3dPipeline.CreateTensorReference<int, Point>(2, new TensorShape(new[] { 1 }));
                vstTimestampPlaceholder1 = map2dTo3dPipeline.CreateTensorReference<int, TimeStamp>(4, new TensorShape(new[] { 1 }));
                vstCameraMatrixPlaceholder1 = map2dTo3dPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { 3, 3 }));
                vstOutputLeftUint8Placeholder1 = map2dTo3dPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
                vstOutputRightUint8Placeholder1 = map2dTo3dPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
                map2dTo3dPositionPlaceholder = map2dTo3dPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));

                // Step 2: Create local tensors
                var pointXYZ = map2dTo3dPipeline.CreateTensor<float, Point>(3, new TensorShape(new[] { 1 }));
                var pintXYZMat = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }));
                var pointXYZMultiplier = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }), 
                    new float[] { 1.0f, -1.0f, 1.0f });
                var offsetTensor = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }), 
                    new float[] { 0.05f, 0.25f, -0.05f });
                var rvecTensor = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }), 
                    new float[] { 0.0f, 0.0f, 0.0f });
                var svecTensor = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }), 
                    new float[] { 0.1f, 0.1f, 0.1f });
                var leftEyeTransform = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));

                // Step 3: Create and connect operators
                var uv2CamOp = map2dTo3dPipeline.CreateOperator<UvTo3DInCameraSpaceOperator>();
                uv2CamOp.SetOperand("uv", uvPlaceholder);
                uv2CamOp.SetOperand("timestamp", vstTimestampPlaceholder1);
                uv2CamOp.SetOperand("camera intrisic", vstCameraMatrixPlaceholder1);
                uv2CamOp.SetOperand("left image", vstOutputLeftUint8Placeholder1);
                uv2CamOp.SetOperand("right image", vstOutputRightUint8Placeholder1);
                uv2CamOp.SetResult("point_xyz", pointXYZ);

                var assignmentOp = map2dTo3dPipeline.CreateOperator<AssignmentOperator>();
                assignmentOp.SetOperand("src", pointXYZ);
                assignmentOp.SetResult("dst", pintXYZMat);

                var elementwiseOp = map2dTo3dPipeline.CreateOperator<ElementwiseMultiplyOperator>();
                elementwiseOp.SetOperand("operand0", pintXYZMat);
                elementwiseOp.SetOperand("operand1", pointXYZMultiplier);
                elementwiseOp.SetResult("result", pintXYZMat);

                var arithmeticOp = map2dTo3dPipeline.CreateOperator<ArithmeticComposeOperator>(
                    new ArithmeticComposeOperatorConfiguration("({0} + {1})"));
                arithmeticOp.SetOperand("{0}", pintXYZMat);
                arithmeticOp.SetOperand("{1}", offsetTensor);
                arithmeticOp.SetResult("result", pintXYZMat);

                var pipelineResult = map2dTo3dPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
                var transformOp = map2dTo3dPipeline.CreateOperator<GetTransformMatrixOperator>();
                transformOp.SetOperand("rotation", rvecTensor);
                transformOp.SetOperand("translation", pintXYZMat);
                transformOp.SetOperand("scale", svecTensor);
                transformOp.SetResult("result", pipelineResult);

                var camSpace2XrLocalOp = map2dTo3dPipeline.CreateOperator<CameraSpaceToWorldOperator>();
                camSpace2XrLocalOp.SetOperand("timestamp", vstTimestampPlaceholder1);
                camSpace2XrLocalOp.SetResult("left", leftEyeTransform);

                var finalArithmeticOp = map2dTo3dPipeline.CreateOperator<ArithmeticComposeOperator>(
                    new ArithmeticComposeOperatorConfiguration("({0} * {1})"));
                finalArithmeticOp.SetOperand("{0}", leftEyeTransform);
                finalArithmeticOp.SetOperand("{1}", pipelineResult);
                finalArithmeticOp.SetResult("result", map2dTo3dPositionPlaceholder);
            }
            catch (Exception e)
            {
                Debug.LogError($"Map2dTo3d pipeline creation failed: {e.Message}");
                mainThreadActions.Enqueue(() => pipelinesReady = false);
            }
        }

        private void CreateRenderPipeline()
        {
            lock(pipelineLock)
            {
                try
                {
                    renderPipeline = provider.CreatePipeline();

                    // Create pipeline placeholders
                    currentPositionRead = renderPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
                    previousPositionRead = renderPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
                    var interpolatedPosition = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));

                    // Create operators
                    var arithmeticOp = renderPipeline.CreateOperator<ArithmeticComposeOperator>(
                        new ArithmeticComposeOperatorConfiguration("({0} * 0.95 + {1} * 0.05)"));
                    var assignmentOp1 = renderPipeline.CreateOperator<AssignmentOperator>();

                    // Connect operators for smooth movement
                    arithmeticOp.SetOperand("{0}", previousPositionRead);
                    arithmeticOp.SetOperand("{1}", currentPositionRead);
                    arithmeticOp.SetResult("result", interpolatedPosition);

                    assignmentOp1.SetOperand("src", interpolatedPosition);
                    assignmentOp1.SetResult("dst", previousPositionRead);

                    // Create GLTF placeholder and position tensor references
                    gltfPlaceholder = renderPipeline.CreateTensorReference<Gltf>();

                    // Create render GLTF operator
                    var renderGltfOperator = renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
                    renderGltfOperator.SetOperand("gltf", gltfPlaceholder);
                    renderGltfOperator.SetOperand("world pose", interpolatedPosition);
                    // renderGltfOperator.SetOperand("is visible", isUfoDetposeTensorectedRead);
                }
                catch (Exception e)
                {
                    Debug.LogError($"Render pipeline creation failed: {e.Message}");
                    mainThreadActions.Enqueue(() => pipelinesReady = false);
                }
            }
        }
        
        private void RunVstPipeline()
        {
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global);
            tensorMapping.Set(vstOutputRightUint8Placeholder, vstOutputRightUint8Global);
            tensorMapping.Set(vstTimestampPlaceholder, vstTimestampGlobal);
            tensorMapping.Set(vstCameraMatrixPlaceholder, vstCameraMatrixGlobal);
            tensorMapping.Set(vstOutputLeftFp32Placeholder, vstOutputLeftFp32Global);

            vstPipeline.Execute(tensorMapping);
        }

        private void RunModelInferencePipeline()
        {
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(vstImagePlaceholder, vstOutputLeftFp32Global);
            tensorMapping.Set(leftEyeUVPlaceholder, leftEyeUVGlobal);
            // tensorMapping.Set(isUfoDetectedWrite, isUfoDetectedGlobal);

            modelInferencePipeline.Execute(tensorMapping);
        }

        private void RunMap2dTo3dPipeline()
        {
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(uvPlaceholder, leftEyeUVGlobal);
            tensorMapping.Set(vstOutputLeftUint8Placeholder1, vstOutputLeftUint8Global);
            tensorMapping.Set(vstOutputRightUint8Placeholder1, vstOutputRightUint8Global);
            tensorMapping.Set(vstTimestampPlaceholder1, vstTimestampGlobal);
            tensorMapping.Set(vstCameraMatrixPlaceholder1, vstCameraMatrixGlobal);
            tensorMapping.Set(map2dTo3dPositionPlaceholder, currentPositionGlobal);

            map2dTo3dPipeline.Execute(tensorMapping);
        }

        private void RunRenderPipeline()
        {
            var tensorMapping = new TensorMapping();
            tensorMapping.Set(previousPositionRead, previousPositionGlobal);
            tensorMapping.Set(currentPositionRead, currentPositionGlobal);
            tensorMapping.Set(gltfPlaceholder, gltfTensor);
            // tensorMapping.Set(isUfoDetectedRead, isUfoDetectedGlobal);

            renderPipeline.Execute(tensorMapping);
        }
    }
}