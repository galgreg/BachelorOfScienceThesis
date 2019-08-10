using System.Collections.Generic;
using UnityEngine;

public class SensorsKit : MonoBehaviour {
    private void Start() {
        ANGLE_BETWEEN_SENSORS = CURRENT_FOV / (RAYS_COUNT - 1);
        STARTING_ANGLE = (MAX_POSSIBLE_FOV - CURRENT_FOV) / 2;
        mTransformComputer =
                new SensorPropertiesComputer(
                        mCarTransform,
                        MAX_SENSOR_LENGTH * UNITY_ANTIBUG_FACTOR,
                        SENSORS_OFFSET_Y,
                        SENSORS_OFFSET_Z);
        InitSensorList();
    }
    private void Update() {
        for (int i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i + STARTING_ANGLE;
            Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
            Vector3 sensorDirection =
                    mTransformComputer.ComputeSensorDirection(currentSensorAngle);
            mSensorList[i].SetRayProperties(sensorOrigin, sensorDirection);
            mSensorList[i].Render();
        }
        List<float> distanceList = GetNormalizedDistanceList();
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
        var newSensor = new CarSensor(
                MAX_SENSOR_LENGTH * UNITY_ANTIBUG_FACTOR,
                UNITY_ANTIBUG_FACTOR);
        Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
        Vector3 sensorDirection =
                mTransformComputer.ComputeSensorDirection(aCurrentSensorAngle);
        newSensor.SetRayProperties(sensorOrigin, sensorDirection);
        return newSensor;
    }

    private List<float> GetNormalizedDistanceList() {
        List<float> distanceList = new List<float>();
        foreach (var sensor in mSensorList) {
            float tempDistance = sensor.GetNormalizedDistance();
            distanceList.Add(tempDistance);
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
    public float CURRENT_FOV = 180.0f;
    public uint RAYS_COUNT = 7;

    private const uint UNITY_ANTIBUG_FACTOR = 100;
    private const float MAX_POSSIBLE_FOV = 180.0f;
    private float ANGLE_BETWEEN_SENSORS;
    private float STARTING_ANGLE;
    private List<CarSensor> mSensorList;
    private SensorPropertiesComputer mTransformComputer;
}
