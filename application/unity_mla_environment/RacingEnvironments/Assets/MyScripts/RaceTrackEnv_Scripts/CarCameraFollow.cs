using UnityEngine;

public class CarCameraFollow : MonoBehaviour
{
    public Transform mCarTransform;
    
    private void Update() {
          Vector3 newCameraPosition = mCarTransform.position;
          newCameraPosition.y = 1.0f;
          transform.position = newCameraPosition;
    }
}
