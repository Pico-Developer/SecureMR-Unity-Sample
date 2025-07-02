using UnityEngine;
using Unity.XR.PXR;


public class TurnOnVST : MonoBehaviour
{
    private void Awake()
    {
        PXR_Manager.EnableVideoSeeThrough = true;
    }
}
