using System;
using System.Text;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;
using Color = Unity.XR.PXR.SecureMR.Color;

namespace PicoXR.SecureMR.Demo
{
    public class MnistwildDemo : MonoBehaviour
    {
        public TextAsset mnistModel;
        public TextAsset tvGltfAsset;
        public int numFramesToRun = -1;
        public float intervalBetweenPipelineRuns = 0.033f;
        private int vstWidth = 3248;
        private int vstHeight = 2464;
        private int cropWidth = 224;
        private int cropHeight = 224;
        private int cropX1 = 1444;
        private int cropY1 = 1332;
        private int cropX2 = 2045;
        private int cropY2 = 1933;

        private int countFrames = 0;
        private float _elapsedTime = 0f;
        private Provider provider;
        private Pipeline pipeline;
        private Pipeline rendererPipeline;

        // Global tensors
        private Tensor predClassGlobal;
        private Tensor predScoreGlobal;
        private Tensor cropRgbGlobal;

        // Pipeline tensors
        private Tensor predClassWrite;
        private Tensor predScoreWrite;
        private Tensor cropRgbWrite;

        // Renderer pipeline tensors
        private Tensor predClassRead;
        private Tensor predScoreRead;
        private Tensor cropRgbRead;

        // GLTF tensors
        private Tensor gltfTensor;
        private Tensor gltfTensor2;
        private Tensor gltfTensor3;
        private Tensor gltfPlaceholder;
        private Tensor gltfPlaceholder2;
        private Tensor gltfPlaceholder3;

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            CreateProvider();
            CreatePipeline();
            CreateRenderer();
        }

        private void Update()
        {
            if (numFramesToRun < 0 || countFrames < numFramesToRun)
            {
                _elapsedTime += Time.deltaTime;
                if (_elapsedTime > intervalBetweenPipelineRuns)
                {
                    RunPipeline();
                    _elapsedTime = 0.0f;
                }
                RenderFrame();
                countFrames++;
            }
        }

        private void CreateProvider()
        {
            provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateMnistModel(Pipeline pipelineParam, Tensor inputTensor, Tensor predClass,
            Tensor predScore)
        {
            // Create MNIST model operator
            var modelConfig = new ModelOperatorConfiguration(mnistModel.bytes, SecureMRModelType.QnnContextBinary, "mnist");
            modelConfig.AddInputMapping("input_1", "input_1", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("_538", "_538", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("_539", "_539", SecureMRModelEncoding.Int32);
            var modelOp = pipelineParam.CreateOperator<RunModelInferenceOperator>(modelConfig);

            modelOp.SetOperand("input_1", inputTensor);
            modelOp.SetResult("_538", predScore);
            modelOp.SetResult("_539", predClass);
        }

        private void CreatePipeline()
        {
            if (provider == null)
            {
                Debug.LogError("Provider is not initialized.");
                return;
            }

            Debug.Log("Creating pipeline...");

            // Create the pipeline
            pipeline = provider.CreatePipeline();

            // Create operators
            var vstOp = pipeline.CreateOperator<RectifiedVstAccessOperator>();
            var getAffineOp = pipeline.CreateOperator<GetAffineOperator>();
            var applyAffineOp = pipeline.CreateOperator<ApplyAffineOperator>();
            var rgbToGrayOp =
                pipeline.CreateOperator<ConvertColorOperator>(
                    new ColorConvertOperatorConfiguration(7)); // 7 = RGB to Gray
            var uint8ToFloat32Op = pipeline.CreateOperator<AssignmentOperator>();
            var normalizeOp =
                pipeline.CreateOperator<ArithmeticComposeOperator>(
                    new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));

            // Create tensors
            var cropShape = new TensorShape(new[] { cropWidth, cropHeight });
            var rawRgb = pipeline.CreateTensor<byte, Matrix>(3, new TensorShape(new[] { vstHeight, vstWidth }));
            var cropGray = pipeline.CreateTensor<byte, Matrix>(1, cropShape);
            var cropGrayFloat =
                pipeline.CreateTensor<float, Matrix>(1, cropShape);
            cropRgbWrite = pipeline.CreateTensorReference<byte, Matrix>(3, cropShape);

            // Create source and destination points for affine transform
            var srcPoints = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 3 }),
                new float[] { cropX1, cropY1, cropX2, cropY1, cropX2, cropY2 });
            var dstPoints = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 3 }),
                new float[] { 0, 0, cropWidth, 0, cropWidth, cropHeight });
            var affineMat = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 2, 3 }));

            // Create MNIST model and tensors
            var inputTensor = pipeline.CreateTensor<float, Matrix>(1, cropShape);
            predClassWrite = pipeline.CreateTensorReference<int, Scalar>(1, new TensorShape(new[]{1}));
            predScoreWrite = pipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(new[]{1}));

            CreateMnistModel(pipeline, inputTensor,predClassWrite, predScoreWrite);
            
            // Create global tensors
            predClassGlobal = provider.CreateTensor<int, Scalar>(1, new TensorShape(new[] { 1 }));
            predScoreGlobal = provider.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
            cropRgbGlobal = provider.CreateTensor<byte, Matrix>(3, cropShape);

        

            // Connect the operators
            vstOp.SetResult("left image", rawRgb);

            getAffineOp.SetOperand("src", srcPoints);
            getAffineOp.SetOperand("dst", dstPoints);
            getAffineOp.SetResult("result", affineMat);

            applyAffineOp.SetOperand("affine", affineMat);
            applyAffineOp.SetOperand("src image", rawRgb);
            applyAffineOp.SetResult("dst image", cropRgbWrite);

            rgbToGrayOp.SetOperand("src", cropRgbWrite);
            rgbToGrayOp.SetResult("dst", cropGray);

            uint8ToFloat32Op.SetOperand("src", cropGray);
            uint8ToFloat32Op.SetResult("dst", cropGrayFloat);

            normalizeOp.SetOperand("{0}", cropGrayFloat);
            normalizeOp.SetResult("result", inputTensor);
        
            Debug.Log("Pipeline created successfully.");
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
            var renderTextOp = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));
            var renderGltfOp = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var renderTextOp2 = rendererPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1440, 960));
            var renderGltfOp2 = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var renderGltfOp3 = rendererPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            var loadTextureOp = rendererPipeline.CreateOperator<LoadTextureOperator>();
            var materialBaseColorTextureConfig = new UpdateGltfOperatorConfiguration(SecureMRGltfOperatorAttribute.MaterialBaseColorTexture);
            var updateGltfOp = rendererPipeline.CreateOperator<UpdateGltfOperator>(materialBaseColorTextureConfig);

            // Create tensors
            var text = rendererPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 30 }),
                Encoding.UTF8.GetBytes("MNIST Demo"));
            var startPosition = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 }); // white text, black background
            var textureId = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize = rendererPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
                new float[] { 144.0f });
            var poseMat = rendererPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, -0.5f,
                    0.0f, 0.5f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            var text2 = rendererPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 30 }),
                Encoding.UTF8.GetBytes("Score"));
            var startPosition2 = rendererPipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }),
                new float[] { 0.1f, 0.3f });
            var colors2 = rendererPipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }),
                new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });
            var textureId2 = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var fontSize2 = rendererPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }),
                new float[] { 144.0f });
            var poseMat2 = rendererPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, 0.5f,
                    0.0f, 0.5f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            var poseMat3 = rendererPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }),
                new float[]
                {
                    0.5f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.5f, 0.0f, 1.0f,
                    0.0f, 0.0f, 0.5f, -1.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                });

            // Create placeholder tensors
            gltfPlaceholder = rendererPipeline.CreateTensorReference<Gltf>();
            gltfPlaceholder2 = rendererPipeline.CreateTensorReference<Gltf>();
            gltfPlaceholder3 = rendererPipeline.CreateTensorReference<Gltf>();

            predClassRead = rendererPipeline.CreateTensorReference<int, Scalar>(1, new TensorShape(new[] {1}));
            predScoreRead = rendererPipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(new[] {1}));
            cropRgbRead = rendererPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(new[] {cropWidth, cropHeight}));

            var gltfMaterialIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }),
                new ushort[] { 0 });
            var gltfTextureIndex = rendererPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }));

            // Create global GLTF tensors
            gltfTensor = provider.CreateTensor<Gltf>(tvGltfAsset.bytes);
            gltfTensor2 = provider.CreateTensor<Gltf>(tvGltfAsset.bytes);
            gltfTensor3 = provider.CreateTensor<Gltf>(tvGltfAsset.bytes);

            // Connect the operators
            renderTextOp.SetOperand("text", predClassRead);
            renderTextOp.SetOperand("start", startPosition);
            renderTextOp.SetOperand("colors", colors);
            renderTextOp.SetOperand("texture ID", textureId);
            renderTextOp.SetOperand("font size", fontSize);
            renderTextOp.SetOperand("gltf", gltfPlaceholder);

            renderGltfOp.SetOperand("gltf", gltfPlaceholder);
            renderGltfOp.SetOperand("world pose", poseMat);
            renderGltfOp.SetOperand("view locked", poseMat);

            renderTextOp2.SetOperand("text", predScoreRead);
            renderTextOp2.SetOperand("start", startPosition2);
            renderTextOp2.SetOperand("colors", colors2);
            renderTextOp2.SetOperand("texture ID", textureId2);
            renderTextOp2.SetOperand("font size", fontSize2);
            renderTextOp2.SetOperand("gltf", gltfPlaceholder2);

            renderGltfOp2.SetOperand("gltf", gltfPlaceholder2);
            renderGltfOp2.SetOperand("world pose", poseMat2);
            renderGltfOp2.SetOperand("view locked", poseMat2);

            loadTextureOp.SetOperand("rgb image", cropRgbRead);
            loadTextureOp.SetOperand("gltf", gltfPlaceholder3);
            loadTextureOp.SetResult("texture ID", gltfTextureIndex);

            updateGltfOp.SetOperand("gltf", gltfPlaceholder3);
            updateGltfOp.SetOperand("material ID", gltfMaterialIndex);
            updateGltfOp.SetOperand("value", gltfTextureIndex);

            renderGltfOp3.SetOperand("gltf", gltfPlaceholder3);
            renderGltfOp3.SetOperand("world pose", poseMat3);
            renderGltfOp3.SetOperand("view locked", poseMat3);

            Debug.Log("Renderer created successfully.");
        }

        private void RunPipeline()
        {
            Debug.Log("Running pipeline...");

            var tensorMapping = new TensorMapping();
            tensorMapping.Set(predClassWrite, predClassGlobal);
            tensorMapping.Set(predScoreWrite, predScoreGlobal);
            tensorMapping.Set(cropRgbWrite, cropRgbGlobal);

            pipeline.Execute(tensorMapping);
        }

        private void RenderFrame()
        {
            Debug.Log("Rendering frame...");

            var tensorMapping = new TensorMapping();
            tensorMapping.Set(gltfPlaceholder, gltfTensor);
            tensorMapping.Set(gltfPlaceholder2, gltfTensor2);
            tensorMapping.Set(gltfPlaceholder3, gltfTensor3);
            tensorMapping.Set(predClassRead, predClassGlobal);
            tensorMapping.Set(predScoreRead, predScoreGlobal);
            tensorMapping.Set(cropRgbRead, cropRgbGlobal);

            rendererPipeline.Execute(tensorMapping);
        }
    }
}