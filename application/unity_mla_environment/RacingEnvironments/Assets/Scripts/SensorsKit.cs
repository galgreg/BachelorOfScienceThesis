using System;
using System.Collections.Generic;
using UnityEngine;

public class SensorsKit : MonoBehaviour {
    private void Start() {
        mTransformComputer =
                new SensorPropertiesComputer(
                        mCarTransform,
                        MAX_SENSOR_LENGTH,
                        SENSORS_OFFSET_Y,
                        SENSORS_OFFSET_Z);
        InitSensorList();
    }
    private void Update() {
        for (int i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i;
            Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
            Vector3 sensorDirection =
                    mTransformComputer.ComputeSensorDirection(currentSensorAngle);
            mSensorList[i].SetRayProperties(sensorOrigin, sensorDirection);
            mSensorList[i].Render();
        }
        List<float> distanceList = GetDistanceList();
        string distanceString = ConvertListToString(distanceList);
        print(distanceString);
    }
    private void InitSensorList() {
        mSensorList = new List<CarSensor>();
        for (uint i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i;
            var newSensor = CreateNewSensor(currentSensorAngle);
            mSensorList.Add(newSensor);
        }
    }
    private CarSensor CreateNewSensor(float aCurrentSensorAngle) {
        var newSensor = new CarSensor(MAX_SENSOR_LENGTH);
        Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
        Vector3 sensorDirection =
                mTransformComputer.ComputeSensorDirection(aCurrentSensorAngle);
        newSensor.SetRayProperties(sensorOrigin, sensorDirection);
        return newSensor;
    }

    private List<float> GetDistanceList() {
        List<float> distanceList = new List<float>();
        foreach (var sensor in mSensorList) {
            distanceList.Add(sensor.GetDistance());
        }
        return distanceList;
    }
    private string ConvertListToString(List<float> aDistanceList) {
        string distanceString = "Distance list: [";
        foreach (var distance in aDistanceList) {
            distanceString += distance.ToString() + ", ";
        }
        distanceString += "]";
        return distanceString;
    }

    public Transform mCarTransform;
    public float MAX_SENSOR_LENGTH = 0.5f;
    public float SENSORS_OFFSET_Y = 0.03f;
    public float SENSORS_OFFSET_Z = 0.026f;

    private const float FIELD_OF_VIEW = 180.0f;
    private const uint RAYS_COUNT = 3;
    private const float ANGLE_BETWEEN_SENSORS = FIELD_OF_VIEW / (RAYS_COUNT - 1);
    private List<CarSensor> mSensorList;
    private SensorPropertiesComputer mTransformComputer;
}
