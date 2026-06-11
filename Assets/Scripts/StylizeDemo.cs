using System;
using System.Collections.Generic;
using ByteDance.PICO.XR;
using ByteDance.PICO.SecureMR;
using UnityEngine;
using UnityEngine.InputSystem;

namespace PicoXR.SecureMR.Demo
{
    public class StylizeDemo : MonoBehaviour
    {
        [Header("Assets")]
        public TextAsset portalGltfAsset;
        public TextAsset stylePredictorModel;
        public TextAsset styleTransferModel;
        public Texture2D[] styleTextures;

        [Header("Scene")]
        public Camera xrCamera;
        public Transform leftPortalTransform;
        public Transform rightPortalTransform;

        [Header("Runtime")]
        public int vstWidth = 3248;
        public int vstHeight = 2464;
        public float intervalBetweenPipelineRuns = 0.033f;
        public ushort portalMaterialId = 1;
        public int styleImageDim = 256;
        public int contentImageDim = 384;

        [Header("Model IO")]
        public string stylePredictorInputNodeName = "style_image";
        public string stylePredictorInputOperatorIOName = "style_image";
        public string stylePredictorOutputNodeName = "mobilenet_conv_Conv_BiasAdd";
        public string stylePredictorOutputOperatorIOName = "mobilenet_conv_Conv_BiasAdd";

        public string styleTransferContentInputNodeName = "content_image";
        public string styleTransferContentInputOperatorIOName = "content_image";
        public string styleTransferStyleInputNodeName = "mobilenet_conv_Conv_BiasAdd";
        public string styleTransferStyleInputOperatorIOName = "mobilenet_conv_Conv_BiasAdd";
        public string styleTransferOutputNodeName = "transformer_expand_conv3_conv_Sigmoid";
        public string styleTransferOutputOperatorIOName = "transformer_expand_conv3_conv_Sigmoid";

        private Provider m_provider;

        private Pipeline m_vstPipeline;
        private Pipeline m_stylePredictionPipeline;
        private Pipeline m_styleTransferPipeline;
        private Pipeline m_renderPipeline;

        private Tensor m_vstLeftGlobal;
        private Tensor m_vstLeftWrite;

        private Tensor[] m_styleTextureGlobal;
        private Tensor[] m_predictedStyleGlobal;
        private Tensor[] m_portalPoseGlobal;
        private Tensor[] m_srcPointsGlobal;
        private Tensor[] m_stylizedRgbGlobal;
        private Tensor[] m_portalGltfGlobal;

        private Tensor m_styleTexturePlaceholder;
        private Tensor m_predictedStylePlaceholder;

        private Tensor m_vstLeftPlaceholder;
        private Tensor m_predictedStyleInputPlaceholder;
        private Tensor m_srcPointsPlaceholder;
        private Tensor m_stylizedRgbWrite;
        private Tensor m_viewLockedGlobal;
        private Tensor m_viewLockedPlaceholder;

        private Tensor[] m_portalGltfPlaceholder;
        private Tensor[] m_portalPosePlaceholder;
        private Tensor[] m_stylizedRgbPlaceholder;

        private int[] m_styleTextureIndex;
        private bool[] m_styleNeedsPrediction;
        private float m_elapsedTime;
        private ulong m_lastPipelineRunId;
        private bool m_leftTriggerPressed;
        private bool m_rightTriggerPressed;
        private InputAction m_leftTriggerAction;
        private InputAction m_rightTriggerAction;

        private const float k_portalPlaneDepthInViewSpace = -2.0f;
        private const float k_portalVerticalOffsetInViewSpace = 0.8f;

        private static readonly Vector3[] k_portalCornerLocal =
        {
            new(-0.51f, 0.68f, 0.0f),
            new(0.51f, 0.68f, 0.0f),
            new(-0.51f, -0.68f, 0.0f),
        };

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            if (xrCamera == null)
            {
                xrCamera = Camera.main;
            }

            if (portalGltfAsset == null || stylePredictorModel == null || styleTransferModel == null)
            {
                Debug.LogError("StylizeDemo missing required assets.");
                enabled = false;
                return;
            }

            if (styleTextures == null || styleTextures.Length == 0)
            {
                Debug.LogError("StylizeDemo requires at least one style texture.");
                enabled = false;
                return;
            }

            CreateProvider();
            CreateGlobals();
            CreatePipelines();

            m_styleTextureIndex = new[] { 0, Mathf.Min(1, styleTextures.Length - 1) };
            m_styleNeedsPrediction = new[] { true, true };
            UploadStyleTexture(0);
            UploadStyleTexture(1);
        }

        private void Update()
        {
            if (m_provider == null)
            {
                return;
            }

            UpdateControllerInput();

            UpdatePortalTensors();

            m_elapsedTime += Time.deltaTime;
            if (m_elapsedTime < intervalBetweenPipelineRuns)
            {
                return;
            }

            m_elapsedTime = 0.0f;
            RunFrame();
        }

        private void OnDestroy()
        {
            DisableInputActions();
            m_vstPipeline?.Destroy();
            m_stylePredictionPipeline?.Destroy();
            m_styleTransferPipeline?.Destroy();
            m_renderPipeline?.Destroy();

            if (m_portalGltfGlobal != null)
            {
                for (int i = 0; i < m_portalGltfGlobal.Length; i++)
                {
                    m_portalGltfGlobal[i]?.Destroy();
                }
            }

            m_vstLeftGlobal?.Destroy();

            if (m_styleTextureGlobal != null)
            {
                for (int i = 0; i < m_styleTextureGlobal.Length; i++)
                {
                    m_styleTextureGlobal[i]?.Destroy();
                    m_predictedStyleGlobal[i]?.Destroy();
                    m_portalPoseGlobal[i]?.Destroy();
                    m_srcPointsGlobal[i]?.Destroy();
                    m_stylizedRgbGlobal[i]?.Destroy();
                }
            }

            m_provider?.Destroy();
            m_provider = null;
        }

        private void OnEnable()
        {
            EnableInputActions();
        }

        private void OnDisable()
        {
            DisableInputActions();
        }

        public void AdvanceLeftStyle()
        {
            AdvanceStyle(0);
        }

        public void AdvanceRightStyle()
        {
            AdvanceStyle(1);
        }

        public void AdvanceBothStyles()
        {
            AdvanceStyle(0);
            AdvanceStyle(1);
        }

        private void CreateProvider()
        {
            m_provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateGlobals()
        {
            m_vstLeftGlobal = m_provider.CreateTensor<byte, Matrix>(3, new TensorShape(vstHeight, vstWidth));

            m_portalGltfGlobal = new Tensor[2];
            m_styleTextureGlobal = new Tensor[2];
            m_predictedStyleGlobal = new Tensor[2];
            m_portalPoseGlobal = new Tensor[2];
            m_srcPointsGlobal = new Tensor[2];
            m_stylizedRgbGlobal = new Tensor[2];

            for (int i = 0; i < 2; i++)
            {
                m_portalGltfGlobal[i] = m_provider.CreateTensor<Gltf>(portalGltfAsset.bytes);
                m_styleTextureGlobal[i] = m_provider.CreateTensor<byte, Matrix>(3, new TensorShape(styleImageDim, styleImageDim));
                m_predictedStyleGlobal[i] = m_provider.CreateTensor<float, Matrix>(100, new TensorShape(1, 1));
                m_portalPoseGlobal[i] = m_provider.CreateTensor<float, Matrix>(1, new TensorShape(4, 4));
                m_srcPointsGlobal[i] = m_provider.CreateTensor<float, Point>(2, new TensorShape(3));
                m_stylizedRgbGlobal[i] = m_provider.CreateTensor<byte, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));
            }

            m_viewLockedGlobal = m_provider.CreateTensor<byte, Scalar>(1, new TensorShape(1), new byte[] { 1 });
        }

        private void CreatePipelines()
        {
            CreateVstPipeline();
            CreateStylePredictionPipeline();
            CreateStyleTransferPipeline();
            CreateRenderPipeline();
        }

        private void CreateVstPipeline()
        {
            m_vstPipeline = m_provider.CreatePipeline();

            var vstOp = m_vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
            m_vstLeftWrite = m_vstPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(vstHeight, vstWidth));
            vstOp.SetResult("left image", m_vstLeftWrite);
        }

        private void CreateStylePredictionPipeline()
        {
            m_stylePredictionPipeline = m_provider.CreatePipeline();

            m_styleTexturePlaceholder =
                m_stylePredictionPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(styleImageDim, styleImageDim));
            m_predictedStylePlaceholder = m_stylePredictionPipeline.CreateTensorReference<float, Matrix>(100, new TensorShape(1, 1));

            var styleRgbFloat = m_stylePredictionPipeline.CreateTensor<float, Matrix>(3, new TensorShape(styleImageDim, styleImageDim));
            var styleRgbNormalized = m_stylePredictionPipeline.CreateTensor<float, Matrix>(3, new TensorShape(styleImageDim, styleImageDim));

            var byteToFloatOp = m_stylePredictionPipeline.CreateOperator<AssignmentOperator>();
            byteToFloatOp.SetOperand("src", m_styleTexturePlaceholder);
            byteToFloatOp.SetResult("dst", styleRgbFloat);

            var normalizeOp = m_stylePredictionPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            normalizeOp.SetOperand("{0}", styleRgbFloat);
            normalizeOp.SetResult("result", styleRgbNormalized);

            var modelConfig = new ModelOperatorConfiguration(stylePredictorModel.bytes, SecureMRModelType.QnnContextBinary, "style_predictor");
            modelConfig.AddInputMapping(stylePredictorInputNodeName, stylePredictorInputOperatorIOName, SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping(stylePredictorOutputNodeName, stylePredictorOutputOperatorIOName, SecureMRModelEncoding.Float32);

            var modelOp = m_stylePredictionPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
            modelOp.SetOperand(stylePredictorInputOperatorIOName, styleRgbNormalized);
            modelOp.SetResult(stylePredictorOutputOperatorIOName, m_predictedStylePlaceholder);
        }

        private void CreateStyleTransferPipeline()
        {
            m_styleTransferPipeline = m_provider.CreatePipeline();

            m_vstLeftPlaceholder = m_styleTransferPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(vstHeight, vstWidth));
            m_predictedStyleInputPlaceholder = m_styleTransferPipeline.CreateTensorReference<float, Matrix>(100, new TensorShape(1, 1));
            m_srcPointsPlaceholder = m_styleTransferPipeline.CreateTensorReference<float, Point>(2, new TensorShape(3));
            m_stylizedRgbWrite = m_styleTransferPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));

            var dstPoints = m_styleTransferPipeline.CreateTensor<float, Point>(2, new TensorShape(3), new[]
            {
                0.0f, 0.0f,
                (float)contentImageDim, 0.0f,
                0.0f, (float)contentImageDim,
            });
            var affineMat = m_styleTransferPipeline.CreateTensor<float, Matrix>(1, new TensorShape(2, 3));
            var slicedRgb = m_styleTransferPipeline.CreateTensor<byte, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));
            var slicedRgbFloat = m_styleTransferPipeline.CreateTensor<float, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));
            var normalizedRgb = m_styleTransferPipeline.CreateTensor<float, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));
            var stylizedFloat = m_styleTransferPipeline.CreateTensor<float, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));
            var stylizedScaled = m_styleTransferPipeline.CreateTensor<float, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));

            var getAffineOp = m_styleTransferPipeline.CreateOperator<GetAffineOperator>();
            getAffineOp.SetOperand("src", m_srcPointsPlaceholder);
            getAffineOp.SetOperand("dst", dstPoints);
            getAffineOp.SetResult("result", affineMat);

            var applyAffineOp = m_styleTransferPipeline.CreateOperator<ApplyAffineOperator>();
            applyAffineOp.SetOperand("affine", affineMat);
            applyAffineOp.SetOperand("src image", m_vstLeftPlaceholder);
            applyAffineOp.SetResult("dst image", slicedRgb);

            var uint8ToFloatOp = m_styleTransferPipeline.CreateOperator<AssignmentOperator>();
            uint8ToFloatOp.SetOperand("src", slicedRgb);
            uint8ToFloatOp.SetResult("dst", slicedRgbFloat);

            var normalizeOp = m_styleTransferPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            normalizeOp.SetOperand("{0}", slicedRgbFloat);
            normalizeOp.SetResult("result", normalizedRgb);

            var modelConfig = new ModelOperatorConfiguration(styleTransferModel.bytes, SecureMRModelType.QnnContextBinary, "style_transfer");
            modelConfig.AddInputMapping(styleTransferContentInputNodeName, styleTransferContentInputOperatorIOName, SecureMRModelEncoding.Float32);
            modelConfig.AddInputMapping(styleTransferStyleInputNodeName, styleTransferStyleInputOperatorIOName, SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping(styleTransferOutputNodeName, styleTransferOutputOperatorIOName, SecureMRModelEncoding.Float32);

            var modelOp = m_styleTransferPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
            modelOp.SetOperand(styleTransferContentInputOperatorIOName, normalizedRgb);
            modelOp.SetOperand(styleTransferStyleInputOperatorIOName, m_predictedStyleInputPlaceholder);
            modelOp.SetResult(styleTransferOutputOperatorIOName, stylizedFloat);

            var scaleOp = m_styleTransferPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} * 255.0"));
            scaleOp.SetOperand("{0}", stylizedFloat);
            scaleOp.SetResult("result", stylizedScaled);

            var floatToUint8Op = m_styleTransferPipeline.CreateOperator<AssignmentOperator>();
            floatToUint8Op.SetOperand("src", stylizedScaled);
            floatToUint8Op.SetResult("dst", m_stylizedRgbWrite);
        }

        private void CreateRenderPipeline()
        {
            m_renderPipeline = m_provider.CreatePipeline();

            m_portalGltfPlaceholder = new Tensor[2];
            m_portalPosePlaceholder = new Tensor[2];
            m_stylizedRgbPlaceholder = new Tensor[2];

            for (int i = 0; i < 2; i++)
            {
                m_portalGltfPlaceholder[i] = m_renderPipeline.CreateTensorReference<Gltf>();
                m_portalPosePlaceholder[i] = m_renderPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(4, 4));
                m_stylizedRgbPlaceholder[i] = m_renderPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(contentImageDim, contentImageDim));
            }

            m_viewLockedPlaceholder = m_renderPipeline.CreateTensorReference<byte, Scalar>(1, new TensorShape(1));

            for (int i = 0; i < 2; i++)
            {
                var loadTextureOp = m_renderPipeline.CreateOperator<LoadTextureOperator>();
                var updateGltfOp = m_renderPipeline.CreateOperator<UpdateGltfOperator>(
                    new UpdateGltfOperatorConfiguration(SecureMRGltfOperatorAttribute.MaterialBaseColorTexture));
                var renderGltfOp = m_renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();

                var materialId = m_renderPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(1), new[] { portalMaterialId });
                var textureId = m_renderPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(1));

                loadTextureOp.SetOperand("rgb image", m_stylizedRgbPlaceholder[i]);
                loadTextureOp.SetOperand("gltf", m_portalGltfPlaceholder[i]);
                loadTextureOp.SetResult("texture ID", textureId);

                updateGltfOp.SetOperand("gltf", m_portalGltfPlaceholder[i]);
                updateGltfOp.SetOperand("material ID", materialId);
                updateGltfOp.SetOperand("value", textureId);

                renderGltfOp.SetOperand("gltf", m_portalGltfPlaceholder[i]);
                renderGltfOp.SetOperand("world pose", m_portalPosePlaceholder[i]);
                renderGltfOp.SetOperand("view locked", m_viewLockedPlaceholder);
            }
        }

        private void AdvanceStyle(int side)
        {
            if (side < 0 || side > 1)
            {
                return;
            }

            if (styleTextures == null || styleTextures.Length == 0)
            {
                return;
            }

            m_styleTextureIndex[side]++;
            if (m_styleTextureIndex[side] >= styleTextures.Length)
            {
                m_styleTextureIndex[side] = 0;
            }

            UploadStyleTexture(side);
            m_styleNeedsPrediction[side] = true;
        }

        private void UploadStyleTexture(int side)
        {
            if (side < 0 || side > 1)
            {
                return;
            }

            Texture2D texture = styleTextures[Mathf.Clamp(m_styleTextureIndex[side], 0, styleTextures.Length - 1)];
            if (!TryGetRgb24(texture, styleImageDim, styleImageDim, out byte[] rgb))
            {
                rgb = CreateFallbackStyleRgb(styleImageDim, styleImageDim);
            }

            m_styleTextureGlobal[side].Reset(rgb);
        }

        private void UpdateControllerInput()
        {
            float leftValue = m_leftTriggerAction != null ? m_leftTriggerAction.ReadValue<float>() : 0.0f;
            float rightValue = m_rightTriggerAction != null ? m_rightTriggerAction.ReadValue<float>() : 0.0f;

            bool leftPressed = leftValue >= 0.75f;
            bool rightPressed = rightValue >= 0.75f;

            if (leftPressed && !m_leftTriggerPressed)
            {
                AdvanceStyle(0);
            }

            if (rightPressed && !m_rightTriggerPressed)
            {
                AdvanceStyle(1);
            }

            m_leftTriggerPressed = leftPressed;
            m_rightTriggerPressed = rightPressed;
        }

        private void EnableInputActions()
        {
            if (m_leftTriggerAction == null)
            {
                m_leftTriggerAction = new InputAction(type: InputActionType.Value, binding: "<XRController>{LeftHand}/trigger");
            }

            if (m_rightTriggerAction == null)
            {
                m_rightTriggerAction = new InputAction(type: InputActionType.Value, binding: "<XRController>{RightHand}/trigger");
            }

            m_leftTriggerAction.Enable();
            m_rightTriggerAction.Enable();
        }

        private void DisableInputActions()
        {
            m_leftTriggerAction?.Disable();
            m_rightTriggerAction?.Disable();
        }

        private void UpdatePortalTensors()
        {
            if (rightPortalTransform == null)
            {
                return;
            }

            UpdatePortalTensor(1, rightPortalTransform, -0.8f);

            if (leftPortalTransform != null)
            {
                UpdatePortalTensor(0, leftPortalTransform, 0.2f);
            }
        }

        private void UpdatePortalTensor(int side, Transform portalTransform, float xOffsetInViewSpace)
        {
            if (xrCamera == null)
            {
                return;
            }

            Vector3 controllerViewPosition = xrCamera.worldToCameraMatrix.MultiplyPoint3x4(portalTransform.position);
            float z = controllerViewPosition.z;
            float t = Mathf.Abs(z) > 1e-5f ? k_portalPlaneDepthInViewSpace / z : 1.0f;

            Vector3 portalViewPosition = controllerViewPosition * t;
            portalViewPosition.x += xOffsetInViewSpace;
            portalViewPosition.y += k_portalVerticalOffsetInViewSpace;

            m_portalPoseGlobal[side].Reset(BuildViewLockedPose(portalViewPosition));
            m_srcPointsGlobal[side].Reset(BuildSrcPointsFromViewPose(portalViewPosition));
        }

        private float[] BuildSrcPointsFromViewPose(Vector3 portalViewPosition)
        {
            Matrix4x4 localToView = Matrix4x4.TRS(portalViewPosition, Quaternion.identity, Vector3.one);
            Matrix4x4 projection = GetLeftEyeProjectionMatrix();

            float[] result = new float[6];
            for (int i = 0; i < 3; i++)
            {
                Vector3 viewCorner = localToView.MultiplyPoint3x4(k_portalCornerLocal[i]);
                Vector4 clipCorner = projection * new Vector4(viewCorner.x, viewCorner.y, viewCorner.z, 1.0f);
                float inverseW = Mathf.Abs(clipCorner.w) > 1e-5f ? 1.0f / clipCorner.w : 1.0f;
                float ndcX = clipCorner.x * inverseW;
                float ndcY = clipCorner.y * inverseW;

                float x = (ndcX + 1.0f) * 0.5f * vstWidth;
                float y = (1.0f - ndcY) * 0.5f * vstHeight;

                result[i * 2 + 0] = x;
                result[i * 2 + 1] = y;
            }

            return result;
        }

        private void RunFrame()
        {
            m_lastPipelineRunId = RunVst(m_lastPipelineRunId);

            for (int side = 0; side < 2; side++)
            {
                if (m_styleNeedsPrediction[side])
                {
                    m_lastPipelineRunId = RunStylePrediction(m_lastPipelineRunId, side);
                    m_styleNeedsPrediction[side] = false;
                }
            }

            for (int side = 0; side < 2; side++)
            {
                m_lastPipelineRunId = RunStyleTransfer(m_lastPipelineRunId, side);
            }

            m_lastPipelineRunId = RunRender(m_lastPipelineRunId);
        }

        private ulong RunVst(ulong pre)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_vstLeftWrite, m_vstLeftGlobal);
            return pre == 0 ? m_vstPipeline.Execute(mapping) : m_vstPipeline.ExecuteAfter(pre, mapping);
        }

        private ulong RunStylePrediction(ulong pre, int side)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_styleTexturePlaceholder, m_styleTextureGlobal[side]);
            mapping.Set(m_predictedStylePlaceholder, m_predictedStyleGlobal[side]);
            return pre == 0 ? m_stylePredictionPipeline.Execute(mapping) : m_stylePredictionPipeline.ExecuteAfter(pre, mapping);
        }

        private ulong RunStyleTransfer(ulong pre, int side)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_vstLeftPlaceholder, m_vstLeftGlobal);
            mapping.Set(m_predictedStyleInputPlaceholder, m_predictedStyleGlobal[side]);
            mapping.Set(m_srcPointsPlaceholder, m_srcPointsGlobal[side]);
            mapping.Set(m_stylizedRgbWrite, m_stylizedRgbGlobal[side]);
            return pre == 0 ? m_styleTransferPipeline.Execute(mapping) : m_styleTransferPipeline.ExecuteAfter(pre, mapping);
        }

        private ulong RunRender(ulong pre)
        {
            var mapping = new TensorMapping();
            for (int side = 0; side < 2; side++)
            {
                mapping.Set(m_portalGltfPlaceholder[side], m_portalGltfGlobal[side]);
                mapping.Set(m_portalPosePlaceholder[side], m_portalPoseGlobal[side]);
                mapping.Set(m_stylizedRgbPlaceholder[side], m_stylizedRgbGlobal[side]);
            }

            mapping.Set(m_viewLockedPlaceholder, m_viewLockedGlobal);

            return pre == 0 ? m_renderPipeline.Execute(mapping) : m_renderPipeline.ExecuteAfter(pre, mapping);
        }

        private static float[] BuildViewLockedPose(Vector3 viewPosition)
        {
            return new[]
            {
                1.0f, 0.0f, 0.0f, viewPosition.x,
                0.0f, 1.0f, 0.0f, viewPosition.y,
                0.0f, 0.0f, 1.0f, viewPosition.z,
                0.0f, 0.0f, 0.0f, 1.0f,
            };
        }

        private Matrix4x4 GetLeftEyeProjectionMatrix()
        {
            if (xrCamera != null && xrCamera.stereoEnabled)
            {
                return xrCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
            }

            return xrCamera != null ? xrCamera.projectionMatrix : Matrix4x4.identity;
        }

        private static bool TryGetRgb24(Texture2D texture, int targetWidth, int targetHeight, out byte[] rgb)
        {
            rgb = null;
            if (texture == null)
            {
                return false;
            }

            Color32[] pixels;
            try
            {
                pixels = texture.GetPixels32();
            }
            catch (Exception)
            {
                return false;
            }

            if (pixels == null || pixels.Length == 0)
            {
                return false;
            }

            int srcWidth = texture.width;
            int srcHeight = texture.height;

            Color32[] resized;
            if (srcWidth == targetWidth && srcHeight == targetHeight)
            {
                resized = pixels;
            }
            else
            {
                resized = new Color32[targetWidth * targetHeight];
                for (int y = 0; y < targetHeight; y++)
                {
                    int srcY = (y * srcHeight) / targetHeight;
                    for (int x = 0; x < targetWidth; x++)
                    {
                        int srcX = (x * srcWidth) / targetWidth;
                        resized[y * targetWidth + x] = pixels[srcY * srcWidth + srcX];
                    }
                }
            }

            rgb = new byte[targetWidth * targetHeight * 3];
            for (int i = 0; i < resized.Length; i++)
            {
                int baseIndex = i * 3;
                rgb[baseIndex + 0] = resized[i].r;
                rgb[baseIndex + 1] = resized[i].g;
                rgb[baseIndex + 2] = resized[i].b;
            }

            return true;
        }

        private static byte[] CreateFallbackStyleRgb(int width, int height)
        {
            byte[] rgb = new byte[width * height * 3];
            for (int i = 0; i < rgb.Length; i += 3)
            {
                rgb[i + 0] = 128;
                rgb[i + 1] = 64;
                rgb[i + 2] = 192;
            }

            return rgb;
        }
    }
}
