using System;
using System.Collections.Generic;
using UnityEngine;

public class SensorsKit : MonoBehaviour {
    private void Start() {
        mTransformComputer =
                new SensorPropertiesComputer(
                        mCarTransform,
                        SENSORS_OFFSET_Y,
                        SENSORS_OFFSET_Z);
        InitSensorList();
    }
    private void InitSensorList() {
        const float ANGLE_BETWEEN_SENSORS = FIELD_OF_VIEW / (RAYS_COUNT - 1);
        for (uint i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i;
            var newSensor = CreateNewSensor(currentSensorAngle);
            mSensorList.Add(newSensor);
        }
    }
    private CarSensor CreateNewSensor(float aCurrentSensorAngle) {
        var newSensor = new CarSensor();
        Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
        Vector3 sensorDirection =
                mTransformComputer.ComputeSensorDirection(aCurrentSensorAngle);
        newSensor.SetRayProperties(sensorOrigin, sensorDirection);
        return newSensor;
    }

    private List<double> GetDistanceList() {
        List<double> distanceList = new List<double>();
        foreach (var sensor in mSensorList) {
            distanceList.Add(sensor.GetDistance());
        }
        return distanceList;
    }

    public Transform mCarTransform;
    public float SENSORS_OFFSET_Y = 0.03f;
    public float SENSORS_OFFSET_Z = 0.026f;

    private const float FIELD_OF_VIEW = 180.0f;
    private const uint RAYS_COUNT = 3;
    private List<CarSensor> mSensorList;
    private SensorPropertiesComputer mTransformComputer;
}
