using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleCarSensors : MonoBehaviour
{
    public void GetSensorsInput() {
        
    }
    private void FixedUpdate() {
        GetSensorsInput();
    }

    public GameObject agentCar;
    public float sensorLength = 2.0f;
}
