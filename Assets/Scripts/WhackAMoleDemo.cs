using System;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using UnityEngine;

namespace PicoXR.SecureMR.Demo
{
    public class WhackAMoleDemo : MonoBehaviour
    {
        public TextAsset detectionModel;
        public TextAsset landmarkModel;
        public TextAsset jointsTransformerJs;
        public TextAsset anchorMat;
        public TextAsset poseMarkerGltf;
        public TextAsset scoreboardGltf;
        public TextAsset duckGltf;
        public TextAsset redboxGltf;

        public int vstWidth = 512;
        public int vstHeight = 512;
        public int roiSize = 256;
        public int detectionInputSize = 128;
        public int maxMoles = 5;
        public int maxJoints = 4;
        public int nodeCount = 13;

        private Provider m_provider;
        private Pipeline m_vstPipeline;
        private Pipeline m_detectionPipeline;
        private Pipeline m_landmarkPipeline;
        private Pipeline m_renderPipeline;

        private Tensor m_vstLeftGlobal;
        private Tensor m_vstLeftWrite;
        private Tensor m_resizedLeftGlobal;

        private Tensor m_isPoseDetectedGlobal;
        private Tensor m_roiAffineGlobal;
        private Tensor m_roiAffineUpdatedGlobal;
        private Tensor m_roiImageGlobal;

        private Tensor m_bodyLandmarkGlobal;
        private Tensor m_molesTransformsGlobal;
        private Tensor m_moleCollisionsGlobal;
        private Tensor m_molesVisibleGlobal;
        private Tensor m_moleDeadlinesGlobal;
        private Tensor m_scoreGlobal;
        private Tensor m_realtimeSinceStartupGlobal;
        private Tensor m_moleOnTimingsGlobal;
        private Tensor m_moleOffTimingsGlobal;

        private Tensor m_poseMarkerGltfGlobal;
        private Tensor m_scoreboardGltfGlobal;
        private Tensor[] m_duckGltfGlobal;
        private Tensor[] m_redboxGltfGlobal;

        private Tensor m_vstLeftPlaceholder;
        private Tensor m_resizedLeftPlaceholder;
        private Tensor m_isPoseDetectedPlaceholder;
        private Tensor m_roiAffinePlaceholder;
        private Tensor m_roiAffineUpdatedPlaceholder;

        private Tensor m_roiImagePlaceholder;
        private Tensor m_bodyLandmarkPlaceholder;
        private Tensor m_moleCollisionsPlaceholder;
        private Tensor m_molesVisiblePlaceholder;
        private Tensor m_moleDeadlinesPlaceholder;
        private Tensor m_scorePlaceholder;
        private Tensor m_realtimeSinceStartupPlaceholder;
        private Tensor m_moleOnTimingsPlaceholder;
        private Tensor m_moleOffTimingsPlaceholder;

        private Tensor m_poseMarkerGltfPlaceholder;
        private Tensor m_scoreboardGltfPlaceholder;
        private Tensor[] m_duckGltfPlaceholder;
        private Tensor[] m_redboxGltfPlaceholder;

        private float m_elapsedTime;
        private ulong m_lastPipelineRunId;
        private System.Random m_random;

        private float[] m_stagePoseData =
        {
            1.0f, 0.0f, 0.0f, 0.6f,
            0.0f, 1.0f, 0.0f, -0.2f,
            0.0f, 0.0f, 1.0f, -2.5f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        private void Awake()
        {
            PXR_Manager.EnableVideoSeeThrough = true;
        }

        private void Start()
        {
            if (detectionModel == null || landmarkModel == null || jointsTransformerJs == null || anchorMat == null)
            {
                Debug.LogError("WhackAMoleDemo missing required assets.");
                enabled = false;
                return;
            }

            if (poseMarkerGltf == null || scoreboardGltf == null || duckGltf == null || redboxGltf == null)
            {
                Debug.LogError("WhackAMoleDemo missing glTF assets.");
                enabled = false;
                return;
            }

            m_random = new System.Random();

            CreateProvider();
            CreateGlobals();
            CreatePipelines();
        }

        private void Update()
        {
            if (m_provider == null)
            {
                return;
            }

            UpdateTimings();
            RunFrame();
        }

        private void OnDestroy()
        {
            m_vstPipeline?.Destroy();
            m_detectionPipeline?.Destroy();
            m_landmarkPipeline?.Destroy();
            m_renderPipeline?.Destroy();

            m_poseMarkerGltfGlobal?.Destroy();
            m_scoreboardGltfGlobal?.Destroy();

            if (m_duckGltfGlobal != null)
            {
                for (int i = 0; i < m_duckGltfGlobal.Length; i++)
                {
                    m_duckGltfGlobal[i]?.Destroy();
                }
            }

            if (m_redboxGltfGlobal != null)
            {
                for (int i = 0; i < m_redboxGltfGlobal.Length; i++)
                {
                    m_redboxGltfGlobal[i]?.Destroy();
                }
            }

            m_vstLeftGlobal?.Destroy();
            m_resizedLeftGlobal?.Destroy();
            m_isPoseDetectedGlobal?.Destroy();
            m_roiAffineGlobal?.Destroy();
            m_roiAffineUpdatedGlobal?.Destroy();
            m_roiImageGlobal?.Destroy();
            m_bodyLandmarkGlobal?.Destroy();
            m_molesTransformsGlobal?.Destroy();
            m_moleCollisionsGlobal?.Destroy();
            m_molesVisibleGlobal?.Destroy();
            m_moleDeadlinesGlobal?.Destroy();
            m_scoreGlobal?.Destroy();
            m_realtimeSinceStartupGlobal?.Destroy();
            m_moleOnTimingsGlobal?.Destroy();
            m_moleOffTimingsGlobal?.Destroy();

            m_provider?.Destroy();
            m_provider = null;
        }

        private void CreateProvider()
        {
            m_provider = new Provider(vstWidth, vstHeight);
        }

        private void CreateGlobals()
        {
            m_vstLeftGlobal = m_provider.CreateTensor<byte, Matrix>(3, new TensorShape(vstHeight, vstWidth));
            m_resizedLeftGlobal = m_provider.CreateTensor<float, Matrix>(3, new TensorShape(detectionInputSize, detectionInputSize));

            m_isPoseDetectedGlobal = m_provider.CreateTensor<byte, Scalar>(1, new TensorShape(1), new byte[] { 1 });
            m_roiAffineGlobal = m_provider.CreateTensor<float, Matrix>(1, new TensorShape(2, 3));
            m_roiAffineUpdatedGlobal = m_provider.CreateTensor<float, Matrix>(1, new TensorShape(2, 3), new[] { 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f });
            m_roiImageGlobal = m_provider.CreateTensor<byte, Matrix>(3, new TensorShape(roiSize, roiSize));

            m_bodyLandmarkGlobal = m_provider.CreateTensor<float, Matrix>(1, new TensorShape(nodeCount, 4, 4));
            m_molesTransformsGlobal = m_provider.CreateTensor<float, Matrix>(1, new TensorShape(maxMoles, 4, 4));
            m_moleCollisionsGlobal = m_provider.CreateTensor<float, Matrix>(1, new TensorShape(maxMoles, 4, 4));
            m_molesVisibleGlobal = m_provider.CreateTensor<byte, Scalar>(1, new TensorShape(maxMoles));
            m_moleDeadlinesGlobal = m_provider.CreateTensor<float, Scalar>(1, new TensorShape(maxMoles));
            m_scoreGlobal = m_provider.CreateTensor<int, Scalar>(1, new TensorShape(1), new[] { 0 });
            m_realtimeSinceStartupGlobal = m_provider.CreateTensor<float, Scalar>(1, new TensorShape(1), new[] { 0.0f });
            m_moleOnTimingsGlobal = m_provider.CreateTensor<float, Scalar>(1, new TensorShape(maxMoles));
            m_moleOffTimingsGlobal = m_provider.CreateTensor<float, Scalar>(1, new TensorShape(maxMoles));

            m_poseMarkerGltfGlobal = m_provider.CreateTensor<Gltf>(poseMarkerGltf.bytes);
            m_scoreboardGltfGlobal = m_provider.CreateTensor<Gltf>(scoreboardGltf.bytes);

            m_duckGltfGlobal = new Tensor[maxMoles];
            for (int i = 0; i < maxMoles; i++)
            {
                m_duckGltfGlobal[i] = m_provider.CreateTensor<Gltf>(duckGltf.bytes);
            }

            m_redboxGltfGlobal = new Tensor[maxJoints];
            for (int i = 0; i < maxJoints; i++)
            {
                m_redboxGltfGlobal[i] = m_provider.CreateTensor<Gltf>(redboxGltf.bytes);
            }

            InitializeMoles();
            UpdateTimings();
        }

        private void InitializeMoles()
        {
            float[] initTransforms = new float[maxMoles * 16];
            for (int i = 0; i < maxMoles; i++)
            {
                int baseIndex = i * 16;
                initTransforms[baseIndex + 0] = 1.0f;
                initTransforms[baseIndex + 5] = 1.0f;
                initTransforms[baseIndex + 10] = 1.0f;
                initTransforms[baseIndex + 15] = 1.0f;
                initTransforms[baseIndex + 3] = -1.0f + i * 0.5f;
                initTransforms[baseIndex + 11] = -2.0f;
            }

            m_molesTransformsGlobal.Reset(initTransforms);
            m_moleCollisionsGlobal.Reset(initTransforms);

            byte[] visible = new byte[maxMoles];
            m_molesVisibleGlobal.Reset(visible);

            float[] deadlines = new float[maxMoles];
            for (int i = 0; i < maxMoles; i++)
            {
                deadlines[i] = 10.0f + (float)m_random.NextDouble() * 4.0f;
            }

            m_moleDeadlinesGlobal.Reset(deadlines);
        }

        private void UpdateTimings()
        {
            float now = Time.time;
            m_realtimeSinceStartupGlobal.Reset(new[] { now });

            float[] onTimings = new float[maxMoles];
            float[] offTimings = new float[maxMoles];
            for (int i = 0; i < maxMoles; i++)
            {
                onTimings[i] = 2.0f + (float)m_random.NextDouble() * 3.0f;
                offTimings[i] = 300.0f + (float)m_random.NextDouble() * 12.0f;
            }

            m_moleOnTimingsGlobal.Reset(onTimings);
            m_moleOffTimingsGlobal.Reset(offTimings);
        }

        private void CreatePipelines()
        {
            CreateVstPipeline();
            CreateDetectionPipeline();
            CreateLandmarkPipeline();
            CreateRenderPipeline();
        }

        private void CreateVstPipeline()
        {
            m_vstPipeline = m_provider.CreatePipeline();

            var vstOp = m_vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
            m_vstLeftWrite = m_vstPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(vstHeight, vstWidth));
            vstOp.SetResult("left image", m_vstLeftWrite);

            float scaleX = (float)detectionInputSize / vstWidth;
            float scaleY = (float)detectionInputSize / vstHeight;
            var resizeMat = m_vstPipeline.CreateTensor<float, Matrix>(1, new TensorShape(2, 3), new[]
            {
                scaleX, 0.0f, 0.0f,
                0.0f, scaleY, 0.0f
            });

            var resizedUint8 = m_vstPipeline.CreateTensor<byte, Matrix>(3, new TensorShape(detectionInputSize, detectionInputSize));
            var resizedFloat = m_vstPipeline.CreateTensor<float, Matrix>(3, new TensorShape(detectionInputSize, detectionInputSize));
            m_resizedLeftPlaceholder = m_vstPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(detectionInputSize, detectionInputSize));

            var applyAffineOp = m_vstPipeline.CreateOperator<ApplyAffineOperator>();
            applyAffineOp.SetOperand("affine", resizeMat);
            applyAffineOp.SetOperand("src image", m_vstLeftWrite);
            applyAffineOp.SetResult("dst image", resizedUint8);

            var uint8ToFloatOp = m_vstPipeline.CreateOperator<AssignmentOperator>();
            uint8ToFloatOp.SetOperand("src", resizedUint8);
            uint8ToFloatOp.SetResult("dst", resizedFloat);

            var normalizeOp = m_vstPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
            normalizeOp.SetOperand("{0}", resizedFloat);
            normalizeOp.SetResult("result", m_resizedLeftPlaceholder);
        }

        private void CreateDetectionPipeline()
        {
            m_detectionPipeline = m_provider.CreatePipeline();

            m_resizedLeftPlaceholder = m_detectionPipeline.CreateTensorReference<float, Matrix>(3, new TensorShape(detectionInputSize, detectionInputSize));
            m_roiAffinePlaceholder = m_detectionPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(2, 3));
            m_isPoseDetectedPlaceholder = m_detectionPipeline.CreateTensorReference<byte, Scalar>(1, new TensorShape(1));

            var poseAnchor = m_detectionPipeline.CreateTensor<float, Matrix>(1, new TensorShape(896, 12));
            var poseScores = m_detectionPipeline.CreateTensor<float, Matrix>(1, new TensorShape(896, 1));

            var modelConfig = new ModelOperatorConfiguration(detectionModel.bytes, SecureMRModelType.QnnContextBinary, "pose");
            modelConfig.AddInputMapping("image", "image", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("box_coords", "pose_anchor", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("box_scores", "score", SecureMRModelEncoding.Float32);

            var modelOp = m_detectionPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
            modelOp.SetOperand("image", m_resizedLeftPlaceholder);
            modelOp.SetResult("pose_anchor", poseAnchor);
            modelOp.SetResult("score", poseScores);

            var identityAffine = m_detectionPipeline.CreateTensor<float, Matrix>(1, new TensorShape(2, 3), new[]
            {
                0.5f, 0.0f, 0.0f,
                0.0f, 0.5f, 0.0f
            });

            var assignAffineOp = m_detectionPipeline.CreateOperator<AssignmentOperator>();
            assignAffineOp.SetOperand("src", identityAffine);
            assignAffineOp.SetResult("dst", m_roiAffinePlaceholder);

            var assignPoseDetected = m_detectionPipeline.CreateOperator<AssignmentOperator>();
            assignPoseDetected.SetOperand("src", m_isPoseDetectedPlaceholder);
            assignPoseDetected.SetResult("dst", m_isPoseDetectedPlaceholder);
        }

        private void CreateLandmarkPipeline()
        {
            m_landmarkPipeline = m_provider.CreatePipeline();

            m_vstLeftPlaceholder = m_landmarkPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(vstHeight, vstWidth));
            m_roiAffineUpdatedPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(2, 3));
            m_roiImagePlaceholder = m_landmarkPipeline.CreateTensorReference<byte, Matrix>(3, new TensorShape(roiSize, roiSize));

            m_bodyLandmarkPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(nodeCount, 4, 4));
            m_moleCollisionsPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Matrix>(1, new TensorShape(maxMoles, 4, 4));
            m_molesVisiblePlaceholder = m_landmarkPipeline.CreateTensorReference<byte, Scalar>(1, new TensorShape(maxMoles));
            m_moleDeadlinesPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(maxMoles));
            m_scorePlaceholder = m_landmarkPipeline.CreateTensorReference<int, Scalar>(1, new TensorShape(1));
            m_realtimeSinceStartupPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(1));
            m_moleOnTimingsPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(maxMoles));
            m_moleOffTimingsPlaceholder = m_landmarkPipeline.CreateTensorReference<float, Scalar>(1, new TensorShape(maxMoles));

            var roiImageFloat = m_landmarkPipeline.CreateTensor<float, Matrix>(3, new TensorShape(roiSize, roiSize));
            var roiImageNormalized = m_landmarkPipeline.CreateTensor<float, Matrix>(3, new TensorShape(roiSize, roiSize));
            var landmarks = m_landmarkPipeline.CreateTensor<float, Matrix>(1, new TensorShape(39, 3));

            var applyAffineOp = m_landmarkPipeline.CreateOperator<ApplyAffineOperator>();
            applyAffineOp.SetOperand("affine", m_roiAffineUpdatedPlaceholder);
            applyAffineOp.SetOperand("src image", m_vstLeftPlaceholder);
            applyAffineOp.SetResult("dst image", m_roiImagePlaceholder);

            var uint8ToFloatOp = m_landmarkPipeline.CreateOperator<AssignmentOperator>();
            uint8ToFloatOp.SetOperand("src", m_roiImagePlaceholder);
            uint8ToFloatOp.SetResult("dst", roiImageFloat);

            var normalizeOp = m_landmarkPipeline.CreateOperator<ArithmeticComposeOperator>(
                new ArithmeticComposeOperatorConfiguration("({0} - 127.5) / 127.5"));
            normalizeOp.SetOperand("{0}", roiImageFloat);
            normalizeOp.SetResult("result", roiImageNormalized);

            var modelConfig = new ModelOperatorConfiguration(landmarkModel.bytes, SecureMRModelType.QnnContextBinary, "pose_landmark");
            modelConfig.AddInputMapping("input_1", "input_1", SecureMRModelEncoding.Float32);
            modelConfig.AddOutputMapping("Identity_4", "landmarks", SecureMRModelEncoding.Float32);

            var modelOp = m_landmarkPipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
            modelOp.SetOperand("input_1", roiImageNormalized);
            modelOp.SetResult("landmarks", landmarks);

            var stagePose = m_landmarkPipeline.CreateTensor<float, Matrix>(1, new TensorShape(4, 4), m_stagePoseData);
            var jsConfig = new JavascriptOperatorConfiguration(jointsTransformerJs.text);
            var jsOp = m_landmarkPipeline.CreateOperator<JavascriptOperator>(jsConfig);
            jsOp.SetOperand("positionArray", landmarks);
            jsOp.SetOperand("moleCollisions", m_moleCollisionsPlaceholder);
            jsOp.SetOperand("realtimesincestartup", m_realtimeSinceStartupPlaceholder);
            jsOp.SetOperand("moleOnTimings", m_moleOnTimingsPlaceholder);
            jsOp.SetOperand("moleOffTimings", m_moleOffTimingsPlaceholder);
            jsOp.SetOperand("moleVisibleArray", m_molesVisiblePlaceholder);
            jsOp.SetOperand("moleDeadlines", m_moleDeadlinesPlaceholder);
            jsOp.SetOperand("score", m_scorePlaceholder);
            jsOp.SetOperand("stagePose", stagePose);
            jsOp.SetResult("matrixArray", m_bodyLandmarkPlaceholder);
            jsOp.SetResult("moleVisibleArray", m_molesVisiblePlaceholder);
            jsOp.SetResult("moleDeadlines", m_moleDeadlinesPlaceholder);
            jsOp.SetResult("score", m_scorePlaceholder);
        }

        private void CreateRenderPipeline()
        {
            m_renderPipeline = m_provider.CreatePipeline();

            m_poseMarkerGltfPlaceholder = m_renderPipeline.CreateTensorReference<Gltf>();
            m_scoreboardGltfPlaceholder = m_renderPipeline.CreateTensorReference<Gltf>();
            m_duckGltfPlaceholder = new Tensor[maxMoles];
            m_redboxGltfPlaceholder = new Tensor[maxJoints];

            for (int i = 0; i < maxMoles; i++)
            {
                m_duckGltfPlaceholder[i] = m_renderPipeline.CreateTensorReference<Gltf>();
            }

            for (int i = 0; i < maxJoints; i++)
            {
                m_redboxGltfPlaceholder[i] = m_renderPipeline.CreateTensorReference<Gltf>();
            }

            var scoreboardPose = m_renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(4, 4), new[]
            {
                0.8f, 0.0f, 0.0f, -0.3f,
                0.0f, 0.8f, 0.0f, 0.2f,
                0.0f, 0.0f, 0.8f, -1.4f,
                0.0f, 0.0f, 0.0f, 1.0f
            });

            var renderScoreboard = m_renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
            renderScoreboard.SetOperand("gltf", m_scoreboardGltfPlaceholder);
            renderScoreboard.SetOperand("world pose", scoreboardPose);

            var scoreTextOp = m_renderPipeline.CreateOperator<RenderTextOperator>(
                new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 256, 256));
            var startPos = m_renderPipeline.CreateTensor<float, Point>(2, new TensorShape(1), new[] { 0.5f, 0.5f });
            var colors = m_renderPipeline.CreateTensor<byte, Unity.XR.PXR.SecureMR.Color>(4, new TensorShape(2), new byte[] { 255, 255, 255, 255, 0, 0, 0, 255 });
            var textureId = m_renderPipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(1), new ushort[] { 0 });
            var fontSize = m_renderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(1), new[] { 64.0f });

            scoreTextOp.SetOperand("text", m_scorePlaceholder);
            scoreTextOp.SetOperand("start", startPos);
            scoreTextOp.SetOperand("colors", colors);
            scoreTextOp.SetOperand("texture ID", textureId);
            scoreTextOp.SetOperand("font size", fontSize);
            scoreTextOp.SetOperand("gltf", m_scoreboardGltfPlaceholder);

            for (int i = 0; i < maxMoles; i++)
            {
                var renderMole = m_renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
                renderMole.SetOperand("gltf", m_duckGltfPlaceholder[i]);
                renderMole.SetOperand("world pose", CreateMolePoseTensor(m_renderPipeline, i));
            }

            for (int i = 0; i < maxJoints; i++)
            {
                var renderRedbox = m_renderPipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
                renderRedbox.SetOperand("gltf", m_redboxGltfPlaceholder[i]);
                renderRedbox.SetOperand("world pose", CreateRedboxPoseTensor(m_renderPipeline, i));
            }
        }

        private Tensor CreateMolePoseTensor(Pipeline pipeline, int index)
        {
            float ox = 0.0f;
            float oy = 0.0f;
            float oz = 0.0f;
            switch (index)
            {
                case 0: ox = -0.4f; oy = 0.5f; break;
                case 1: ox = 0.4f; oy = 0.5f; break;
                case 2: ox = -0.4f; oy = 0.0f; break;
                case 3: ox = 0.4f; oy = 0.0f; break;
                case 4: ox = 0.0f; oy = 0.9f; break;
            }

            float[] pose =
            {
                1.0f, 0.0f, 0.0f, ox,
                0.0f, 1.0f, 0.0f, oy,
                0.0f, 0.0f, 1.0f, oz,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            return pipeline.CreateTensor<float, Matrix>(1, new TensorShape(4, 4), pose);
        }

        private Tensor CreateRedboxPoseTensor(Pipeline pipeline, int index)
        {
            int[] joints = { 2, 3, 8, 9 };
            int jointIndex = joints[Mathf.Clamp(index, 0, joints.Length - 1)];
            float offset = jointIndex * 0.02f;

            float[] pose =
            {
                1.0f, 0.0f, 0.0f, offset,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f
            };

            return pipeline.CreateTensor<float, Matrix>(1, new TensorShape(4, 4), pose);
        }

        private void RunFrame()
        {
            m_lastPipelineRunId = RunVst(m_lastPipelineRunId);
            m_lastPipelineRunId = RunDetection(m_lastPipelineRunId);
            m_lastPipelineRunId = RunLandmark(m_lastPipelineRunId);
            m_lastPipelineRunId = RunRender(m_lastPipelineRunId);
        }

        private ulong RunVst(ulong pre)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_vstLeftWrite, m_vstLeftGlobal);
            mapping.Set(m_resizedLeftPlaceholder, m_resizedLeftGlobal);
            return pre == 0 ? m_vstPipeline.Execute(mapping) : m_vstPipeline.ExecuteAfter(pre, mapping);
        }

        private ulong RunDetection(ulong pre)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_resizedLeftPlaceholder, m_resizedLeftGlobal);
            mapping.Set(m_roiAffinePlaceholder, m_roiAffineGlobal);
            mapping.Set(m_isPoseDetectedPlaceholder, m_isPoseDetectedGlobal);
            return pre == 0 ? m_detectionPipeline.Execute(mapping) : m_detectionPipeline.ExecuteAfter(pre, mapping);
        }

        private ulong RunLandmark(ulong pre)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_vstLeftPlaceholder, m_vstLeftGlobal);
            mapping.Set(m_roiAffineUpdatedPlaceholder, m_roiAffineUpdatedGlobal);
            mapping.Set(m_roiImagePlaceholder, m_roiImageGlobal);
            mapping.Set(m_bodyLandmarkPlaceholder, m_bodyLandmarkGlobal);
            mapping.Set(m_moleCollisionsPlaceholder, m_moleCollisionsGlobal);
            mapping.Set(m_molesVisiblePlaceholder, m_molesVisibleGlobal);
            mapping.Set(m_moleDeadlinesPlaceholder, m_moleDeadlinesGlobal);
            mapping.Set(m_scorePlaceholder, m_scoreGlobal);
            mapping.Set(m_realtimeSinceStartupPlaceholder, m_realtimeSinceStartupGlobal);
            mapping.Set(m_moleOnTimingsPlaceholder, m_moleOnTimingsGlobal);
            mapping.Set(m_moleOffTimingsPlaceholder, m_moleOffTimingsGlobal);
            return pre == 0 ? m_landmarkPipeline.Execute(mapping) : m_landmarkPipeline.ExecuteAfter(pre, mapping);
        }

        private ulong RunRender(ulong pre)
        {
            var mapping = new TensorMapping();
            mapping.Set(m_poseMarkerGltfPlaceholder, m_poseMarkerGltfGlobal);
            mapping.Set(m_scoreboardGltfPlaceholder, m_scoreboardGltfGlobal);

            for (int i = 0; i < maxMoles; i++)
            {
                mapping.Set(m_duckGltfPlaceholder[i], m_duckGltfGlobal[i]);
            }

            for (int i = 0; i < maxJoints; i++)
            {
                mapping.Set(m_redboxGltfPlaceholder[i], m_redboxGltfGlobal[i]);
            }

            return pre == 0 ? m_renderPipeline.Execute(mapping) : m_renderPipeline.ExecuteAfter(pre, mapping);
        }
    }
}
