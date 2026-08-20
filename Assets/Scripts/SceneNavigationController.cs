using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;
using UnityEngine.XR.Interaction.Toolkit.UI;
using ByteDance.PICO.XR;

namespace PicoXR.SecureMR.Demo
{
    [DisallowMultipleComponent]
    public sealed class SceneNavigationController : MonoBehaviour
    {
        private const string k_logTag = "[SceneNavigation]";
        private const string k_loaderScenePath = "Assets/Scenes/SceneLoader.unity";
        private const string k_runtimeThemeResourcePath = "SceneNavigationRuntimeTheme";
        private const float k_panelDistanceInMeters = 1.5f;
        private const float k_panelWorldScale = 0.12f;
        private const float k_navigationThreshold = 0.6f;
        private const float k_navigationRepeatDelayInSeconds = 0.2f;
        private const float k_backButtonLeftInPixels = 32f;
        private const float k_backButtonTopInPixels = 32f;
        private const float k_backButtonWidthInPixels = 260f;
        private const float k_backButtonHeightInPixels = 58f;
        private const int k_panelWidthInPixels = 960;
        private const int k_panelHeightInPixels = 720;

        private static readonly Color s_buttonColor = new(0.18f, 0.28f, 0.52f, 0.98f);
        private static readonly Color s_selectedButtonColor = new(0.28f, 0.48f, 0.88f, 1f);
        private static SceneNavigationController s_instance;

        private readonly List<Button> m_navigationButtons = new(8);
        private readonly List<Action> m_navigationActions = new(8);

        private UIDocument m_uiDocument;
        private PanelSettings m_panelSettings;
        private BoxCollider m_panelCollider;
        private VisualElement m_root;
        private InputAction m_navigateAction;
        private InputAction m_submitAction;
        private InputAction m_cancelAction;
        private InputAction m_systemBackAction;
        private Camera m_mainCamera;
        private bool m_isLoading;
        private bool m_isQuitting;
        private bool m_isApplicationPaused;
        private bool m_shouldLoadLoaderAfterResume;
        private bool m_isNavigationHeld;
        private int m_selectedButtonIndex;
        private float m_nextNavigationTime;
        private Task<int> m_activeCleanupTask;

        private void Awake()
        {
            if (s_instance != null && s_instance != this)
            {
                gameObject.SetActive(false);
                Destroy(gameObject);
                return;
            }

            s_instance = this;
            DontDestroyOnLoad(gameObject);
            CreateDocument();
            CreateInputActions();
            Application.wantsToQuit += HandleApplicationWantsToQuit;
        }

        private void OnEnable()
        {
            SceneManager.sceneLoaded += HandleSceneLoaded;
            PXR_Plugin.System.FocusStateLost += HandlePicoFocusStateLost;
            PXR_Plugin.System.FocusStateAcquired += HandlePicoFocusStateAcquired;
            PXR_Plugin.System.SessionStateChanged += HandlePicoSessionStateChanged;
            m_navigateAction?.Enable();
            m_submitAction?.Enable();
            m_cancelAction?.Enable();
            m_systemBackAction?.Enable();
        }

        private void OnDisable()
        {
            SceneManager.sceneLoaded -= HandleSceneLoaded;
            PXR_Plugin.System.FocusStateLost -= HandlePicoFocusStateLost;
            PXR_Plugin.System.FocusStateAcquired -= HandlePicoFocusStateAcquired;
            PXR_Plugin.System.SessionStateChanged -= HandlePicoSessionStateChanged;
            m_navigateAction?.Disable();
            m_submitAction?.Disable();
            m_cancelAction?.Disable();
            m_systemBackAction?.Disable();
        }

        private void OnDestroy()
        {
            if (s_instance != this)
            {
                return;
            }

            Application.wantsToQuit -= HandleApplicationWantsToQuit;
            m_navigateAction?.Dispose();
            m_submitAction?.Dispose();
            m_cancelAction?.Dispose();
            m_systemBackAction?.Dispose();

            if (m_panelSettings != null)
            {
                Destroy(m_panelSettings);
            }

            s_instance = null;
        }

        private void Update()
        {
            if (m_isLoading || m_navigationButtons.Count == 0)
            {
                return;
            }

            HandleNavigationInput();

            if (m_submitAction.WasPressedThisFrame())
            {
                m_navigationActions[m_selectedButtonIndex].Invoke();
                return;
            }

            if (m_cancelAction.WasPressedThisFrame() &&
                !IsLoaderScene(SceneManager.GetActiveScene()))
            {
                LoadLoaderScene();
                return;
            }

            if (m_systemBackAction.WasPressedThisFrame() &&
                !IsLoaderScene(SceneManager.GetActiveScene()))
            {
                LoadLoaderScene();
            }
        }

        private void LateUpdate()
        {
            PositionPanelInFrontOfCamera();
        }

        private void OnApplicationPause(bool isPaused)
        {
            m_isApplicationPaused = isPaused;
            Scene activeScene = SceneManager.GetActiveScene();
            Debug.Log($"{k_logTag} Application pause changed: isPaused={isPaused}, scene={activeScene.name}");

            if (isPaused)
            {
                PrepareForBackground("Application pause");
                return;
            }

            LoadLoaderAfterBackgroundIfNeeded();
        }

        private void CreateDocument()
        {
            m_panelSettings = ScriptableObject.CreateInstance<PanelSettings>();
            m_panelSettings.name = "Runtime Scene Navigation Panel";
            m_panelSettings.renderMode = PanelRenderMode.WorldSpace;
            m_panelSettings.scaleMode = PanelScaleMode.ConstantPixelSize;
            m_panelSettings.sortingOrder = 1000;
            m_panelSettings.themeStyleSheet =
                Resources.Load<ThemeStyleSheet>(k_runtimeThemeResourcePath);

            if (m_panelSettings.themeStyleSheet == null)
            {
                Debug.LogError(
                    $"Scene navigation theme is missing from Resources: {k_runtimeThemeResourcePath}");
            }

            m_uiDocument = gameObject.AddComponent<UIDocument>();
            m_uiDocument.panelSettings = m_panelSettings;
            m_uiDocument.sortingOrder = 1000;
            m_uiDocument.worldSpaceSizeMode = UIDocument.WorldSpaceSizeMode.Fixed;
            m_uiDocument.worldSpaceSize = new Vector2(k_panelWidthInPixels, k_panelHeightInPixels);

            m_panelCollider = gameObject.AddComponent<BoxCollider>();
            m_panelCollider.isTrigger = true;

            var inputConfiguration = gameObject.AddComponent<PanelInputConfiguration>();
            inputConfiguration.processWorldSpaceInput = true;
            inputConfiguration.panelInputRedirection = PanelInputConfiguration.PanelInputRedirection.Never;
            inputConfiguration.defaultEventCameraIsMainCamera = true;
            inputConfiguration.autoCreatePanelComponents = true;
            inputConfiguration.interactionLayers = ~0;

            gameObject.AddComponent<XRUIToolkitManager>();
            transform.localScale = Vector3.one * k_panelWorldScale;

            m_root = m_uiDocument.rootVisualElement;
            m_root.name = "scene-navigation-root";
            m_root.pickingMode = PickingMode.Ignore;
            m_root.style.position = Position.Absolute;
            m_root.style.left = 0;
            m_root.style.top = 0;
            m_root.style.right = 0;
            m_root.style.bottom = 0;
        }

        private void CreateInputActions()
        {
            m_navigateAction = new InputAction(
                "Scene Navigation",
                InputActionType.Value,
                expectedControlType: "Vector2");
            m_navigateAction.AddBinding("<XRController>{LeftHand}/primary2DAxis");
            m_navigateAction.AddBinding("<XRController>{RightHand}/primary2DAxis");
            m_navigateAction.AddBinding("<Gamepad>/leftStick");
            m_navigateAction.AddCompositeBinding("2DVector")
                .With("Up", "<Keyboard>/upArrow")
                .With("Down", "<Keyboard>/downArrow")
                .With("Left", "<Keyboard>/leftArrow")
                .With("Right", "<Keyboard>/rightArrow");

            m_submitAction = new InputAction("Select Scene", InputActionType.Button);
            m_submitAction.AddBinding("<XRController>{LeftHand}/triggerButton");
            m_submitAction.AddBinding("<XRController>{RightHand}/triggerButton");
            m_submitAction.AddBinding("<XRController>{LeftHand}/primary2DAxisClick");
            m_submitAction.AddBinding("<XRController>{RightHand}/primary2DAxisClick");
            m_submitAction.AddBinding("<Gamepad>/buttonSouth");
            m_submitAction.AddBinding("<Keyboard>/enter");
            m_submitAction.AddBinding("<Keyboard>/numpadEnter");

            m_cancelAction = new InputAction("Back to Scene Loader", InputActionType.Button);
            m_cancelAction.AddBinding("<XRController>{LeftHand}/secondaryButton");
            m_cancelAction.AddBinding("<XRController>{RightHand}/secondaryButton");
            m_cancelAction.AddBinding("<Gamepad>/buttonEast");
            m_cancelAction.AddBinding("<Keyboard>/escape");

            m_systemBackAction = new InputAction("System Back", InputActionType.Button);
            m_systemBackAction.AddBinding("<Keyboard>/escape");
            m_systemBackAction.AddBinding("<XRController>{LeftHand}/menuButton");
            m_systemBackAction.AddBinding("<XRController>{RightHand}/menuButton");
            m_systemBackAction.AddBinding("<PICO4TouchController>{LeftHand}/menu");
            m_systemBackAction.AddBinding("<PICO4UltraController>{LeftHand}/menu");
            m_systemBackAction.AddBinding("<PICONeo3Controller>{LeftHand}/menu");
            m_systemBackAction.AddBinding("<PICOG3TouchController>/menu");
        }

        private bool HandleApplicationWantsToQuit()
        {
            if (m_isQuitting)
            {
                return true;
            }

            Scene activeScene = SceneManager.GetActiveScene();

            if (!IsLoaderScene(activeScene))
            {
                Debug.Log(
                    $"{k_logTag} Quit requested from {activeScene.name}; " +
                    "draining SecureMR owner(s) before returning to SceneLoader.");
                m_shouldLoadLoaderAfterResume = false;
                LoadLoaderScene();
                return false;
            }

            m_isQuitting = true;
            Debug.Log($"{k_logTag} Quit requested from SceneLoader.");
            return true;
        }

        private void HandlePicoFocusStateLost()
        {
            Scene activeScene = SceneManager.GetActiveScene();
            Debug.Log($"{k_logTag} PICO XR focus lost in scene={activeScene.name}");
            PrepareForBackground("PICO XR focus loss");
        }

        private void HandlePicoFocusStateAcquired()
        {
            Scene activeScene = SceneManager.GetActiveScene();
            Debug.Log($"{k_logTag} PICO XR focus acquired in scene={activeScene.name}");

            if (!m_isApplicationPaused)
            {
                LoadLoaderAfterBackgroundIfNeeded();
            }
        }

        private void HandlePicoSessionStateChanged(XrSessionState sessionState)
        {
            Scene activeScene = SceneManager.GetActiveScene();
            Debug.Log(
                $"{k_logTag} PICO XR session changed: " +
                $"state={sessionState}, scene={activeScene.name}");

            if (sessionState == XrSessionState.Focused)
            {
                if (!m_isApplicationPaused)
                {
                    LoadLoaderAfterBackgroundIfNeeded();
                }

                return;
            }

            if (sessionState == XrSessionState.Visible ||
                sessionState == XrSessionState.Stopping ||
                sessionState == XrSessionState.LossPending ||
                sessionState == XrSessionState.Exiting)
            {
                PrepareForBackground($"PICO XR session {sessionState}");
            }
        }

        private void HandleSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            Debug.Log($"{k_logTag} Loaded scene: {scene.name} ({scene.path}), mode={mode}");
            m_isLoading = false;
            m_activeCleanupTask = null;
            m_mainCamera = null;
            RebuildUi(scene);
            PositionPanelInFrontOfCamera();
        }

        private void RebuildUi(Scene activeScene)
        {
            m_root.Clear();
            m_root.SetEnabled(true);
            m_navigationButtons.Clear();
            m_navigationActions.Clear();
            m_selectedButtonIndex = 0;
            m_isNavigationHeld = false;

            if (IsLoaderScene(activeScene))
            {
                ConfigureLoaderCollider();
                BuildSceneList();
                return;
            }

            ConfigureBackButtonCollider();
            BuildBackButton();
        }

        private void BuildSceneList()
        {
            var overlay = new VisualElement
            {
                name = "scene-loader-overlay",
                pickingMode = PickingMode.Position,
            };
            overlay.style.position = Position.Absolute;
            overlay.style.left = 0;
            overlay.style.top = 0;
            overlay.style.right = 0;
            overlay.style.bottom = 0;
            overlay.style.alignItems = Align.Center;
            overlay.style.justifyContent = Justify.Center;
            overlay.style.backgroundColor = new Color(0.025f, 0.035f, 0.07f, 0.94f);
            m_root.Add(overlay);

            var card = new VisualElement
            {
                name = "scene-loader-card",
            };
            card.style.width = 720;
            card.style.maxWidth = Length.Percent(90);
            card.style.maxHeight = Length.Percent(86);
            card.style.paddingLeft = 36;
            card.style.paddingTop = 32;
            card.style.paddingRight = 36;
            card.style.paddingBottom = 32;
            card.style.backgroundColor = new Color(0.07f, 0.09f, 0.16f, 0.98f);
            card.style.borderTopLeftRadius = 24;
            card.style.borderTopRightRadius = 24;
            card.style.borderBottomLeftRadius = 24;
            card.style.borderBottomRightRadius = 24;
            overlay.Add(card);

            var title = new Label("SpatialML Samples");
            title.style.fontSize = 38;
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            title.style.color = Color.white;
            title.style.marginBottom = 8;
            card.Add(title);

            var subtitle = new Label("Select a scene to run");
            subtitle.style.fontSize = 20;
            subtitle.style.color = new Color(0.7f, 0.75f, 0.86f);
            subtitle.style.marginBottom = 24;
            card.Add(subtitle);

            var sceneList = new ScrollView(ScrollViewMode.Vertical)
            {
                name = "build-scene-list",
            };
            sceneList.style.flexGrow = 1;
            card.Add(sceneList);

            Button firstSceneButton = null;
            int sceneCount = SceneManager.sceneCountInBuildSettings;

            for (int buildIndex = 0; buildIndex < sceneCount; buildIndex++)
            {
                string scenePath = SceneUtility.GetScenePathByBuildIndex(buildIndex);
                if (scenePath == k_loaderScenePath)
                {
                    continue;
                }

                int selectedBuildIndex = buildIndex;
                string sceneName = Path.GetFileNameWithoutExtension(scenePath);
                Action loadSelectedScene = () => LoadScene(selectedBuildIndex);
                var sceneButton = CreateButton(sceneName, loadSelectedScene);
                sceneButton.tooltip = scenePath;
                sceneList.Add(sceneButton);
                RegisterNavigationButton(sceneButton, loadSelectedScene);
                firstSceneButton ??= sceneButton;
            }

            if (firstSceneButton == null)
            {
                var emptyMessage = new Label("No additional enabled scenes were found in Build Settings.");
                emptyMessage.style.fontSize = 18;
                emptyMessage.style.color = new Color(1.0f, 0.67f, 0.3f);
                emptyMessage.style.whiteSpace = WhiteSpace.Normal;
                sceneList.Add(emptyMessage);
                return;
            }

            SetSelectedButtonIndex(0);
            firstSceneButton.schedule.Execute(firstSceneButton.Focus);
        }

        private void BuildBackButton()
        {
            var backButton = CreateButton("←  Back to Samples", LoadLoaderScene);
            backButton.name = "back-to-scene-loader";
            backButton.style.position = Position.Absolute;
            backButton.style.left = k_backButtonLeftInPixels;
            backButton.style.top = k_backButtonTopInPixels;
            backButton.style.width = k_backButtonWidthInPixels;
            backButton.style.height = k_backButtonHeightInPixels;
            m_root.Add(backButton);
            RegisterNavigationButton(backButton, LoadLoaderScene);
            SetSelectedButtonIndex(0);
            backButton.schedule.Execute(backButton.Focus);
        }

        private static Button CreateButton(string text, Action clicked)
        {
            var button = new Button(clicked)
            {
                text = text,
            };
            button.style.height = 68;
            button.style.marginBottom = 14;
            button.style.paddingLeft = 22;
            button.style.paddingRight = 22;
            button.style.fontSize = 22;
            button.style.unityFontStyleAndWeight = FontStyle.Bold;
            button.style.unityTextAlign = TextAnchor.MiddleLeft;
            button.style.color = Color.white;
            button.style.backgroundColor = s_buttonColor;
            button.style.borderTopLeftRadius = 12;
            button.style.borderTopRightRadius = 12;
            button.style.borderBottomLeftRadius = 12;
            button.style.borderBottomRightRadius = 12;
            return button;
        }

        private void RegisterNavigationButton(Button button, Action action)
        {
            int buttonIndex = m_navigationButtons.Count;
            m_navigationButtons.Add(button);
            m_navigationActions.Add(action);
            button.RegisterCallback<FocusInEvent>(_ => SetSelectedButtonIndex(buttonIndex));
            button.RegisterCallback<PointerEnterEvent>(_ => SetSelectedButtonIndex(buttonIndex));
        }

        private void HandleNavigationInput()
        {
            float verticalInput = m_navigateAction.ReadValue<Vector2>().y;
            if (Mathf.Abs(verticalInput) < k_navigationThreshold)
            {
                m_isNavigationHeld = false;
                return;
            }

            if (m_isNavigationHeld && Time.unscaledTime < m_nextNavigationTime)
            {
                return;
            }

            int direction = verticalInput > 0f ? -1 : 1;
            int nextButtonIndex = m_selectedButtonIndex + direction;

            if (nextButtonIndex < 0)
            {
                nextButtonIndex = m_navigationButtons.Count - 1;
            }
            else if (nextButtonIndex >= m_navigationButtons.Count)
            {
                nextButtonIndex = 0;
            }

            SetSelectedButtonIndex(nextButtonIndex);
            m_navigationButtons[m_selectedButtonIndex].Focus();
            m_isNavigationHeld = true;
            m_nextNavigationTime = Time.unscaledTime + k_navigationRepeatDelayInSeconds;
        }

        private void SetSelectedButtonIndex(int buttonIndex)
        {
            if (buttonIndex < 0 || buttonIndex >= m_navigationButtons.Count)
            {
                return;
            }

            if (m_selectedButtonIndex < m_navigationButtons.Count)
            {
                m_navigationButtons[m_selectedButtonIndex].style.backgroundColor = s_buttonColor;
            }

            m_selectedButtonIndex = buttonIndex;
            m_navigationButtons[m_selectedButtonIndex].style.backgroundColor = s_selectedButtonColor;
        }

        private void ConfigureLoaderCollider()
        {
            m_panelCollider.center = Vector3.zero;
            m_panelCollider.size = new Vector3(k_panelWidthInPixels, k_panelHeightInPixels, 1f);
        }

        private void ConfigureBackButtonCollider()
        {
            float centerX = -k_panelWidthInPixels * 0.5f +
                            k_backButtonLeftInPixels +
                            k_backButtonWidthInPixels * 0.5f;
            float centerY = k_panelHeightInPixels * 0.5f -
                            k_backButtonTopInPixels -
                            k_backButtonHeightInPixels * 0.5f;
            m_panelCollider.center = new Vector3(centerX, centerY, 0f);
            m_panelCollider.size = new Vector3(
                k_backButtonWidthInPixels,
                k_backButtonHeightInPixels,
                1f);
        }

        private void PositionPanelInFrontOfCamera()
        {
            if (m_mainCamera == null)
            {
                m_mainCamera = Camera.main;
            }

            if (m_mainCamera == null)
            {
                return;
            }

            Transform cameraTransform = m_mainCamera.transform;
            Vector3 panelPosition = cameraTransform.position +
                                    cameraTransform.forward * k_panelDistanceInMeters;
            transform.SetPositionAndRotation(panelPosition, cameraTransform.rotation);
        }

        private void LoadLoaderScene()
        {
            int loaderBuildIndex = FindBuildIndex(k_loaderScenePath);
            if (loaderBuildIndex < 0)
            {
                Debug.LogError($"Scene loader is missing from Build Settings: {k_loaderScenePath}");
                return;
            }

            LoadScene(loaderBuildIndex);
        }

        private async void LoadScene(int buildIndex)
        {
            if (m_isLoading)
            {
                return;
            }

            m_isLoading = true;
            m_root.SetEnabled(false);

            Scene activeScene = SceneManager.GetActiveScene();
            if (!IsLoaderScene(activeScene))
            {
                int cleanupCount = await ShutdownActiveSecureMRServicesAsync();
                Debug.Log(
                    $"{k_logTag} Drained {cleanupCount} SecureMR owner(s) in {activeScene.name}.");
            }

            SceneManager.LoadScene(buildIndex, LoadSceneMode.Single);
        }

        private async void PrepareForBackground(string lifecycleEvent)
        {
            if (m_isLoading ||
                m_shouldLoadLoaderAfterResume ||
                IsLoaderScene(SceneManager.GetActiveScene()))
            {
                return;
            }

            m_shouldLoadLoaderAfterResume = true;
            int cleanupCount = await ShutdownActiveSecureMRServicesAsync();
            Debug.Log(
                $"{k_logTag} {lifecycleEvent} outside SceneLoader; " +
                $"shut down {cleanupCount} SecureMR owner(s).");

            if (!m_isApplicationPaused)
            {
                LoadLoaderAfterBackgroundIfNeeded();
            }
        }

        private void LoadLoaderAfterBackgroundIfNeeded()
        {
            if (!m_shouldLoadLoaderAfterResume)
            {
                return;
            }

            m_shouldLoadLoaderAfterResume = false;
            Debug.Log($"{k_logTag} Returned after sample backgrounding; requesting SceneLoader.");
            LoadLoaderScene();
        }

        private Task<int> ShutdownActiveSecureMRServicesAsync()
        {
            m_activeCleanupTask ??= ShutdownActiveSecureMRServicesCoreAsync();
            return m_activeCleanupTask;
        }

        private static Task<int> ShutdownActiveSecureMRServicesCoreAsync()
        {
            MonoBehaviour[] sceneBehaviours =
                FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            int cleanupCount = 0;

            for (int i = 0; i < sceneBehaviours.Length; i++)
            {
                if (sceneBehaviours[i] is ISecureMRServiceCleanup cleanup)
                {
                    cleanup.ShutdownSecureMR();
                    cleanupCount++;
                }
            }

            return Task.FromResult(cleanupCount);
        }

        private static int FindBuildIndex(string scenePath)
        {
            int sceneCount = SceneManager.sceneCountInBuildSettings;
            for (int buildIndex = 0; buildIndex < sceneCount; buildIndex++)
            {
                if (SceneUtility.GetScenePathByBuildIndex(buildIndex) == scenePath)
                {
                    return buildIndex;
                }
            }

            return -1;
        }

        private static bool IsLoaderScene(Scene scene)
        {
            return scene.path == k_loaderScenePath;
        }
    }
}
