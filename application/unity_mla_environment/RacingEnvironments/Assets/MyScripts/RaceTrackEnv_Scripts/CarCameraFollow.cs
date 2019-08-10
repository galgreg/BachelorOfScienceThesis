using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarCameraFollow : MonoBehaviour
{
    public GameObject agentCar;
    
    private void Update() {
          Vector3 pos = agentCar.transform.position;
          Quaternion rotation = agentCar.transform.rotation;
          pos.y = 1.0f;
          transform.position = pos;
    }
}
